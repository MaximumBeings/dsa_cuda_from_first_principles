# Getting Started

This book targets CUDA C++ compiled with `nvcc` for real NVIDIA GPU architectures — `sm_70` (Volta) and newer covers everything this book builds. A real NVIDIA GPU with a matching driver is what you need to *run* the code in this book; `nvcc` itself only needs a CUDA toolkit and will happily compile for architectures your current machine doesn't have.

## Installing a toolchain

A full NVIDIA CUDA Toolkit install (from developer.nvidia.com) gives you `nvcc`, `ptxas`, and every header this book uses. If you only need to compile and read compiler diagnostics — not run kernels — a much lighter option exists and is exactly what this book's own examples were authored and verified with:

```bash
pip install --break-system-packages nvidia-cuda-nvcc-cu12
```

This installs a genuine, working `nvcc` (bundled CUDA Runtime headers and `libcudart` included) without requiring a GPU, a driver, or a full toolkit install. Every `.cu` file in this book compiles cleanly against it.

```bash
nvcc --version   # compiler, needed to build anything
nvcc -arch=sm_80 hello_cuda.cu -o hello_cuda   # pick the arch flag matching your GPU
```

## The honesty discipline this book follows

This book's reference implementations were authored and compiled in an environment with a full CUDA toolchain (`nvcc`, real CUDA Runtime headers and library) but **no NVIDIA GPU and no NVIDIA driver**. Every kernel in this book is genuinely compiled for a real target architecture — none of that is fabricated. What that environment cannot do is *execute* a kernel. This book handles that limitation the same way throughout, and it is worth stating once, clearly, rather than repeating a caveat on every page:

- **Plain C++ host code** (no `__global__`, no `__device__`, no CUDA Runtime API calls) is genuinely compiled with `g++` and genuinely run. Its output is locked into the page and re-verified by a fresh recompile and rerun before publication.
- **CUDA kernels that need a device to execute** are genuinely compiled with `nvcc` for a real architecture. Since launching them needs hardware this environment does not have, each one is checked instead by a **host-side simulation**: ordinary C++ that walks the exact same grid, block, and thread loop nest the kernel's own launch configuration would produce, applying the identical index arithmetic and boundary checks the kernel source contains, then checked against an independent scalar reference implementation. This is not a substitute for running the real kernel — it is a real, exhaustive verification of the kernel's *logic*, which is where nearly every genuine bug in a data structure's parallel implementation actually lives (an index formula that double-counts a position, a boundary check that guards the wrong axis, a race between two writes) rather than in anything specific to a physical device.
- **CUDA Runtime API calls that can be genuinely made without a device** (`cudaGetDeviceCount`, `cudaMalloc`, `cudaHostAlloc`, and similar) are genuinely called, and this environment honestly reports back whatever a driver-less, device-less machine actually reports — typically `cudaErrorNoDevice` or `cudaErrorInsufficientDriver` — rather than a fabricated success.
- **Timing and throughput numbers are never fabricated.** Where a chapter's argument depends on which of two approaches is faster rather than merely which is correct, this book measures a genuinely computed, deterministic quantity instead of a wall-clock number — a count of memory operations, cache lines, or comparisons — specifically because wall-clock timing captured once on one machine is not reproducible on a rerun, let alone on a reader's own hardware, the way an exact count is.

Every chapter states which of the above applies to each piece of its own code, so nothing is left for a reader to guess about how a claim was actually established.

## Compile-line conventions

- Plain C++ host files (`.cpp`): `g++ -std=c++17 -Wall -Wextra -O2 file.cpp -o binary`
- CUDA files with real device code (`.cu`): `nvcc -arch=sm_80 file.cu -o binary`
- Files that call the CUDA Runtime API from the host add `-lcudart` and the toolchain's include/library paths, shown inline wherever they're used.

## Prerequisites

This book assumes working knowledge of C++ (structs, templates, pointers, RAII) and does not assume any prior CUDA experience — Part 0 builds the execution and memory model this book needs from nothing. It does assume familiarity with basic sequential data structures and Big-O notation at the level of an introductory data structures course; Part 0's own Chapter 3 is what extends that vocabulary to a parallel machine.
