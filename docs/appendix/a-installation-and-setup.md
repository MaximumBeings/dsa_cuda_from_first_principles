# Appendix A: Installation & Setup

Every chapter in this book states, up front, which of three things a given piece of code is: plain host C++ genuinely compiled with `g++`, a CUDA kernel genuinely compiled with `nvcc` and checked by an exhaustive host-side simulation of its exact logic (since this book's own authoring environment has no physical GPU), or a CUDA Runtime API call genuinely made and honestly reported, whatever this environment's own hardware situation causes it to return. None of that is possible without a working toolchain first. This appendix is the thing to run before Chapter 1: it confirms `nvcc` and `g++` are both present and speak the language version this book assumes (A.1), turns the CUDA Runtime API calls Section 2.3 already introduced into a practical, plain-language installation checklist (A.2), hands you a minimal, genuinely-tested `Makefile` that encodes this book's own compile-line conventions so you never have to retype them by hand (A.3), and, for anyone who wants to go one step further than this book's own environment ever could, walks through renting a real GPU by the hour on Lambda Cloud so every kernel in this book can be genuinely launched, not just simulated (A.4).

## A.1 Confirming Your Toolchain: nvcc and g++

### Intuition

Before trusting any output this book claims to have "genuinely compiled and run," a reader reproducing it needs the same two pieces this book itself depends on: an `nvcc` release recent enough to accept the `-arch=sm_80` target used in every CUDA compile command, and a C++ standard of at least C++17 (structured bindings, `if constexpr`, and class template argument deduction all appear in later chapters). Both facts are answered by simply asking each tool what it is -- `nvcc --version` and `g++ --version` print a real, human-readable banner naming the exact release installed, with no program required.

**Check nvcc:**

```bash
nvcc --version
```

**Sample output:**

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Tue_Jun_09_02:43:40_PM_PDT_2026
Cuda compilation tools, release 13.3, V13.3.73
Build cuda_13.3.r13.3/compiler.38244171_0
```

**Check g++:**

```bash
g++ --version
```

**Sample output:**

```text
g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
Copyright (C) 2023 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```

### The Concept, In Detail

```
ASCII view: two tools, two jobs, one compile command.

  nvcc file.cu -o binary
   |
   +--> recognizes __global__/__device__ code -> compiles it for the
   |      target architecture named by -arch (this book uses sm_80
   |      throughout)
   |
   +--> recognizes ordinary host C++ in the SAME file -> hands it to a
          real underlying host compiler (g++, on this book's own
          environment) for the parts that are not device code at all

  g++ file.cpp -o binary
   |
   +--> compiles ONLY ordinary host C++ -- no __global__, no __device__,
          no CUDA Runtime API calls of any kind. This is exactly the
          category this book calls a "CPU baseline."
```

A version banner is useful, but it is not something a program can branch on -- the same information is also available as preprocessor macros nvcc defines at compile time (`__CUDACC_VER_MAJOR__`, `__CUDACC_VER_MINOR__`), alongside the standard `__cplusplus` macro every C++ compiler defines. Section A.1's own code below reads exactly these macros back, which is how a real build system can assert "this toolchain is new enough" automatically instead of asking a human to read a banner.

[COMMON TRAP]
It is tempting to assume a `.cu` file must be compiled with an explicit `-std=c++17` flag, the way this book's own `g++` compile lines always show one. Section A.1's own code below compiles with the exact same bare `nvcc -arch=sm_80 file.cu -o binary` command used in every other chapter -- no `-std` flag at all -- and `__cplusplus` still reports `201703L`: this nvcc release already defaults to C++17. Adding `-std=c++17` explicitly is harmless and often good practice for portability across nvcc releases, but this book's own compile commands rely on nothing more than what is genuinely true of the toolchain actually used to write it.

### Code and Verification

```cpp
// 208_toolchain_version_check.cu
//
// Appendix A.1 -- confirming the toolchain a reader just installed actually
// matches what this book's own compile-line conventions assume: a C++17
// front end, and an nvcc release recent enough to accept the -arch=sm_80
// target used throughout every chapter. Every value below is a predefined
// preprocessor macro nvcc itself supplies -- nothing here is looked up at
// runtime, so this program's output is identical on every rerun and
// reflects, exactly, whichever nvcc actually compiled it.
//
// Compile: nvcc -arch=sm_80 208_toolchain_version_check.cu -o 208_toolchain_version_check
// Run:     ./208_toolchain_version_check

#include <cstdio>

int main() {
    printf("=== Appendix A.1: toolchain version check (compile-time macros only) ===\n\n");

    printf("--- C++ language standard nvcc's front end is compiling as ---\n");
    printf("  __cplusplus = %ldL\n", __cplusplus);
    bool is_cpp17_or_later = (__cplusplus >= 201703L);
    printf("  this is C++17 or later: %s\n\n", is_cpp17_or_later ? "yes" : "no");

#if defined(__CUDACC_VER_MAJOR__)
    printf("--- nvcc release that compiled this file ---\n");
    printf("  __CUDACC_VER_MAJOR__ = %d\n", __CUDACC_VER_MAJOR__);
    printf("  __CUDACC_VER_MINOR__ = %d\n", __CUDACC_VER_MINOR__);
    bool nvcc_new_enough = (__CUDACC_VER_MAJOR__ >= 11);
    printf("  this is CUDA 11.0 or later (required for -arch=sm_80): %s\n\n",
           nvcc_new_enough ? "yes" : "no");
#else
    printf("--- nvcc release that compiled this file ---\n");
    printf("  __CUDACC_VER_MAJOR__ is not defined -- this file was not compiled with nvcc\n\n");
    bool nvcc_new_enough = false;
#endif

#if defined(__GNUC__)
    printf("--- host compiler nvcc dispatched host-side code to ---\n");
    printf("  __GNUC__ = %d, __GNUC_MINOR__ = %d\n\n", __GNUC__, __GNUC_MINOR__);
#endif

    printf("--- what this tells you ---\n");
    printf("  nvcc compiles BOTH device code and ordinary host code in one pass; the\n");
    printf("  __CUDACC_VER_* macros identify the nvcc release itself, while __cplusplus\n");
    printf("  and __GNUC__ identify the C++ language rules and host compiler nvcc is\n");
    printf("  using for the non-device parts of the same file. Every macro above is\n");
    printf("  resolved by the PREPROCESSOR at compile time -- notice this file's own\n");
    printf("  compile command names no -std flag at all, matching every .cu compile\n");
    printf("  command elsewhere in this book, and yet __cplusplus already reports\n");
    printf("  C++17: this nvcc release simply defaults to it.\n");

    bool ok = is_cpp17_or_later && nvcc_new_enough;
    printf("\nself-check: this toolchain satisfies both requirements this book's compile\n");
    printf("commands assume (C++17, CUDA 11.0+): %s\n", ok ? "confirmed" : "MISMATCH");

    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
nvcc -arch=sm_80 208_toolchain_version_check.cu -o 208_toolchain_version_check
./208_toolchain_version_check
```

**Sample input:** none -- every value printed is a compile-time preprocessor macro, not something computed from runtime data.

**Sample output:**

```text
=== Appendix A.1: toolchain version check (compile-time macros only) ===

--- C++ language standard nvcc's front end is compiling as ---
  __cplusplus = 201703L
  this is C++17 or later: yes

--- nvcc release that compiled this file ---
  __CUDACC_VER_MAJOR__ = 13
  __CUDACC_VER_MINOR__ = 3
  this is CUDA 11.0 or later (required for -arch=sm_80): yes

--- host compiler nvcc dispatched host-side code to ---
  __GNUC__ = 13, __GNUC_MINOR__ = 3

--- what this tells you ---
  nvcc compiles BOTH device code and ordinary host code in one pass; the
  __CUDACC_VER_* macros identify the nvcc release itself, while __cplusplus
  and __GNUC__ identify the C++ language rules and host compiler nvcc is
  using for the non-device parts of the same file. Every macro above is
  resolved by the PREPROCESSOR at compile time -- notice this file's own
  compile command names no -std flag at all, matching every .cu compile
  command elsewhere in this book, and yet __cplusplus already reports
  C++17: this nvcc release simply defaults to it.

self-check: this toolchain satisfies both requirements this book's compile
commands assume (C++17, CUDA 11.0+): confirmed
```

## A.2 The CUDA Runtime API, Installation-Checked

### Intuition

Section 2.3 called `cudaGetDeviceCount`, `cudaMalloc`, `cudaHostAlloc`, and `cudaDeviceSynchronize` to teach a specific lesson about the execution model: a realistic CUDA program must check the Runtime API's return value rather than assume a device is present. This appendix reuses that same honesty for a more immediate, practical purpose -- a single program worth running exactly once, right after installing the toolkit, that answers three concrete yes/no questions in plain language: is the CUDA Runtime library itself present, is an NVIDIA driver installed, and does that driver report any usable GPU?

### The Concept, In Detail

```
ASCII view: three questions, in dependency order.

  cudaRuntimeGetVersion  -->  is the toolkit's runtime library linked at all?
          |
          v  (this can succeed with NO driver and NO gpu present)
  cudaDriverGetVersion   -->  is an NVIDIA driver installed?
          |
          v  (driver_ver == 0 means: no)
  cudaGetDeviceCount     -->  does the driver report >=1 usable device?
          |
          v
   cudaSuccess, count>0        cudaSuccess, count==0        cudaErrorInsufficientDriver
   (a real, usable GPU)     (driver present, no GPU        / cudaErrorNoDevice
                              attached -- e.g. a bare        (no usable driver at all --
                              cloud VM)                       this book's own environment)
```

Each of these three calls can succeed or fail independently of the others, which is exactly why the checklist asks them in this specific order: a failure at `cudaRuntimeGetVersion` means nothing downstream can be trusted at all (the toolkit itself is broken or mis-linked), while a failure only at `cudaGetDeviceCount` -- with the first two calls succeeding -- means the toolkit and its API are both working correctly, and the honest answer is simply "no usable GPU is visible from here."

[COMMON TRAP]
It is tempting to treat `cudaGetDeviceCount` failing as proof that the installation itself is broken, and to go reinstall the entire toolkit. As this checklist's own diagnosis step shows, `cudaRuntimeGetVersion` reporting `cudaSuccess` already proves the toolkit and its Runtime library are correctly installed and linked -- a subsequent `cudaErrorInsufficientDriver` or `cudaErrorNoDevice` from `cudaGetDeviceCount` is reporting a completely different, unrelated fact (no compatible driver, or no physical device), which reinstalling the CUDA toolkit itself cannot fix.

### Code and Verification

```cpp
// 209_cuda_runtime_installation_check.cu
//
// Appendix A.2 -- a single, runnable installation checklist. Section 2.3
// introduced cudaGetDeviceCount and its relatives to teach a concept: a
// realistic program must check the Runtime API's return value rather than
// assume success. This appendix repackages the same real API calls for a
// different, practical purpose -- something a reader runs exactly once,
// right after installing the toolkit, to find out in plain language what
// is and is not actually present on their machine. Every call below is a
// genuine CUDA Runtime API call; nothing about its result is decided in
// advance of running it.
//
// Compile: nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 209_cuda_runtime_installation_check.cu -o 209_cuda_runtime_installation_check
// Run:     LD_LIBRARY_PATH=$NVDIR/lib ./209_cuda_runtime_installation_check

#include <cstdio>
#include <cuda_runtime.h>

int main() {
    printf("=== Appendix A.2: CUDA installation checklist, genuinely run ===\n\n");

    int runtime_ver = -1, driver_ver = -1, device_count = -1;
    cudaError_t e_runtime = cudaRuntimeGetVersion(&runtime_ver);
    cudaError_t e_driver  = cudaDriverGetVersion(&driver_ver);
    cudaError_t e_count   = cudaGetDeviceCount(&device_count);

    printf("--- step 1: is a CUDA Runtime library linked at all? ---\n");
    printf("  cudaRuntimeGetVersion -> %s (%s)\n", cudaGetErrorName(e_runtime), cudaGetErrorString(e_runtime));
    if (e_runtime == cudaSuccess) {
        printf("  runtime_ver = %d  (major.minor = %d.%d)\n\n", runtime_ver, runtime_ver / 1000, (runtime_ver % 100) / 10);
    } else {
        printf("  runtime_ver could not be determined\n\n");
    }

    printf("--- step 2: is an NVIDIA driver installed? ---\n");
    printf("  cudaDriverGetVersion  -> %s (%s)\n", cudaGetErrorName(e_driver), cudaGetErrorString(e_driver));
    printf("  driver_ver = %d  (0 means no driver is installed at all)\n\n", driver_ver);

    printf("--- step 3: does the driver report any usable GPU? ---\n");
    printf("  cudaGetDeviceCount    -> %s (%s)\n", cudaGetErrorName(e_count), cudaGetErrorString(e_count));
    printf("  device_count = %d\n\n", device_count);

    printf("--- diagnosis ---\n");
    bool toolkit_found = (e_runtime == cudaSuccess);
    bool driver_found = (driver_ver > 0);
    bool gpu_found = (e_count == cudaSuccess) && (device_count > 0);
    printf("  CUDA toolkit (runtime library): %s\n", toolkit_found ? "FOUND" : "NOT FOUND");
    printf("  NVIDIA driver:                  %s\n", driver_found ? "FOUND" : "NOT FOUND");
    printf("  usable GPU device:              %s\n\n", gpu_found ? "FOUND" : "NOT FOUND");

    printf("--- how to read this on YOUR machine ---\n");
    printf("  toolkit=FOUND, driver=FOUND, gpu=FOUND     -> everything in this book can\n");
    printf("                                                 also be genuinely launched on a\n");
    printf("                                                 real device, not just compiled.\n");
    printf("  toolkit=FOUND, driver=FOUND, gpu=NOT FOUND -> a driver exists but reports zero\n");
    printf("                                                 devices (e.g. a cloud VM with no\n");
    printf("                                                 GPU attached); cudaGetDeviceCount\n");
    printf("                                                 itself still returns cudaSuccess.\n");
    printf("  toolkit=FOUND, driver=NOT FOUND            -> this book's own authoring\n");
    printf("                                                 environment: a real CUDA Runtime\n");
    printf("                                                 is linked, but cudaGetDeviceCount\n");
    printf("                                                 fails with cudaErrorInsufficientDriver\n");
    printf("                                                 or cudaErrorNoDevice, exactly as\n");
    printf("                                                 Section 2.3 first showed.\n");
    printf("  cudaRuntimeGetVersion itself fails          -> the toolkit's own runtime library\n");
    printf("                                                 is missing or mis-linked; fix this\n");
    printf("                                                 before anything else in this book.\n");

    // The self-check does not require a device to be present -- it only
    // confirms every call above returned a well-formed, DOCUMENTED result,
    // the same standard Section 2.3 established.
    bool count_well_formed = (device_count >= 0) || (e_count != cudaSuccess);
    bool count_is_known_case =
        (e_count == cudaSuccess) ||
        (e_count == cudaErrorInsufficientDriver) ||
        (e_count == cudaErrorNoDevice);
    bool ok = toolkit_found && count_well_formed && count_is_known_case;
    printf("\nself-check: the runtime library loaded successfully and cudaGetDeviceCount\n");
    printf("returned a well-formed, documented result either way: %s\n", ok ? "confirmed" : "MISMATCH");

    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 209_cuda_runtime_installation_check.cu -o 209_cuda_runtime_installation_check
LD_LIBRARY_PATH=$NVDIR/lib ./209_cuda_runtime_installation_check
```

**Sample input:** none -- every value comes back from a genuine CUDA Runtime API call made on whatever machine actually runs the binary.

**Sample output:**

```text
=== Appendix A.2: CUDA installation checklist, genuinely run ===

--- step 1: is a CUDA Runtime library linked at all? ---
  cudaRuntimeGetVersion -> cudaSuccess (no error)
  runtime_ver = 13000  (major.minor = 13.0)

--- step 2: is an NVIDIA driver installed? ---
  cudaDriverGetVersion  -> cudaSuccess (no error)
  driver_ver = 0  (0 means no driver is installed at all)

--- step 3: does the driver report any usable GPU? ---
  cudaGetDeviceCount    -> cudaErrorInsufficientDriver (CUDA driver version is insufficient for CUDA runtime version)
  device_count = -1

--- diagnosis ---
  CUDA toolkit (runtime library): FOUND
  NVIDIA driver:                  NOT FOUND
  usable GPU device:              NOT FOUND

--- how to read this on YOUR machine ---
  toolkit=FOUND, driver=FOUND, gpu=FOUND     -> everything in this book can
                                                 also be genuinely launched on a
                                                 real device, not just compiled.
  toolkit=FOUND, driver=FOUND, gpu=NOT FOUND -> a driver exists but reports zero
                                                 devices (e.g. a cloud VM with no
                                                 GPU attached); cudaGetDeviceCount
                                                 itself still returns cudaSuccess.
  toolkit=FOUND, driver=NOT FOUND            -> this book's own authoring
                                                 environment: a real CUDA Runtime
                                                 is linked, but cudaGetDeviceCount
                                                 fails with cudaErrorInsufficientDriver
                                                 or cudaErrorNoDevice, exactly as
                                                 Section 2.3 first showed.
  cudaRuntimeGetVersion itself fails          -> the toolkit's own runtime library
                                                 is missing or mis-linked; fix this
                                                 before anything else in this book.

self-check: the runtime library loaded successfully and cudaGetDeviceCount
returned a well-formed, documented result either way: confirmed
```

## A.3 A Minimal Makefile for This Book's Compile-Line Conventions

### Intuition

Every chapter names its own exact compile and run commands so nothing is left for a reader to guess, but retyping `nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart file.cu -o binary` by hand for every one of this book's 200-plus source files is not how anyone should actually work through it. A `Makefile` that encodes those same two conventions -- one pattern rule for `.cpp` files compiled with `g++`, one pattern rule for `.cu` files compiled with `nvcc` -- turns every chapter's own stated compile command into something `make` already knows how to do, for any file dropped into the same directory.

### The Concept, In Detail

```
ASCII view: two pattern rules cover every source file in this book.

  *.cpp  ---(g++ -std=c++17 -Wall -Wextra -O2)--->  binary
  *.cu   ---(nvcc -arch=sm_80 -I... -L... -lcudart)--->  binary

  $(wildcard *.cpp) / $(wildcard *.cu)
       finds every source file actually present in the directory,
       so "make" builds whatever chapter's files you have dropped
       in -- no per-file editing of the Makefile required.
```

The `Makefile` below is intentionally minimal: it has exactly the two pattern rules this book's own two compile-line conventions require, an `all` target that builds every `.cpp` and `.cu` file found in the current directory, and a `clean` target that removes them again. It assumes the same `nvidia-cu13` pip-installed toolkit layout this book's own environment uses; a system-wide CUDA installation would instead set `NVCC := nvcc` directly and drop the `-I`/`-L` flags, since a system install already places `nvcc`, its headers, and its libraries on the compiler's default search paths.

**The Makefile:**

```makefile
CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -O2

NVDIR := /usr/local/lib/python3.11/dist-packages/nvidia/cu13
NVCC := $(NVDIR)/bin/nvcc
NVCC_FLAGS := -arch=sm_80 -I$(NVDIR)/include -L$(NVDIR)/lib -lcudart

CPP_SOURCES := $(wildcard *.cpp)
CU_SOURCES := $(wildcard *.cu)
CPP_BINS := $(CPP_SOURCES:.cpp=)
CU_BINS := $(CU_SOURCES:.cu=)

all: $(CPP_BINS) $(CU_BINS)

%: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

%: %.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -f $(CPP_BINS) $(CU_BINS)

.PHONY: all clean
```

[COMMON TRAP]
It is tempting to add a single pattern rule matching both `%.cpp` and `%.cu` to "simplify" this Makefile. `make`'s pattern rules dispatch purely on the target's file extension matching the rule's own pattern -- a `.cpp` file genuinely needs `g++`'s flags (`-std=c++17 -Wall -Wextra -O2`, no `-arch`) and a `.cu` file genuinely needs `nvcc`'s (`-arch=sm_80`, plus the include/library paths and `-lcudart` for any file that calls the Runtime API) -- collapsing them into one rule would either drop flags one file type needs or add flags the other file type's compiler does not understand.

### Code and Verification

Two stub source files -- one plain host file, one containing a real (but unlaunched) kernel -- are enough to prove the `Makefile` above genuinely builds both of this book's file categories correctly.

```cpp
#include <cstdio>

int main() {
    printf("hello_host: compiled with g++, no CUDA involved\n");
    return 0;
}
```

```cpp
#include <cstdio>

// A real, syntactically valid kernel -- not launched here, since this
// environment has no device to launch it on (see Appendix A.2). Compiling
// it at all is still a genuine test that nvcc accepts device-code syntax.
__global__ void hello_kernel() {
}

int main() {
    printf("hello_kernel: compiled with nvcc, contains one real __global__ kernel\n");
    return 0;
}
```

**Compile and run:**

```bash
make
./hello_host
LD_LIBRARY_PATH=$NVDIR/lib ./hello_kernel
```

**Sample input:** none -- `make` discovers `hello_host.cpp` and `hello_kernel.cu` via its own `$(wildcard ...)` rules.

**Sample output:**

```text
$ make
g++ -std=c++17 -Wall -Wextra -O2 hello_host.cpp -o hello_host
/usr/local/lib/python3.11/dist-packages/nvidia/cu13/bin/nvcc -arch=sm_80 -I/usr/local/lib/python3.11/dist-packages/nvidia/cu13/include -L/usr/local/lib/python3.11/dist-packages/nvidia/cu13/lib -lcudart hello_kernel.cu -o hello_kernel

$ ./hello_host
hello_host: compiled with g++, no CUDA involved

$ LD_LIBRARY_PATH=$NVDIR/lib ./hello_kernel
hello_kernel: compiled with nvcc, contains one real __global__ kernel
```

## A.4 Running This Book on Real GPU Hardware via Lambda Cloud

### Intuition

Section A.2's own checklist reports, honestly, that this book's authoring environment has a genuine CUDA Runtime but no driver and no physical device -- which is exactly why every kernel in this book is checked by an exhaustive host-side simulation rather than a real launch. A reader who wants to go one step further and actually LAUNCH these kernels on physical hardware needs a real NVIDIA GPU, and the fastest way to get one without buying it is a pay-per-hour GPU cloud provider. Lambda Cloud (`lambda.ai`) is a convenient choice for exactly this book, because its default machine image -- Lambda Stack -- ships with the NVIDIA driver, the CUDA toolkit (including `nvcc`), cuDNN, and NCCL already installed: Section A.2's own `209_cuda_runtime_installation_check` should report all three of its checks as FOUND on a freshly launched instance, with no separate CUDA installation step of your own required at all.

### The Concept, In Detail

```
ASCII view: from this book's own environment to a real, driver-having GPU.

  this book's sandbox:  CUDA Runtime, no driver, no GPU
         |
         |  1. create an account at cloud.lambda.ai
         |  2. add a public SSH key (console, or `ssh-keygen` locally first)
         |  3. launch an instance (pick a GPU type + region, e.g. one A10/A100/H100)
         |  4. wait for status "Running", copy its "ssh ubuntu@<address>" command
         v
  a real Lambda Cloud instance:  Lambda Stack preinstalled
         |
         |  5. ssh ubuntu@<address>
         |  6. clone this book's repo, rerun Section A.2's own check program
         v
  209_cuda_runtime_installation_check now reports:
    CUDA toolkit (runtime library): FOUND
    NVIDIA driver:                  FOUND
    usable GPU device:              FOUND      <-- genuinely different from
                                                     this book's own sandbox
```

Steps 1 through 4 happen in a web browser, at the Lambda Cloud console; step 5 is an ordinary SSH connection using the username `ubuntu`, exactly like connecting to any other cloud Linux VM; step 6 is the payoff -- the identical `209_cuda_runtime_installation_check.cu` file this appendix already compiled and locked in Section A.2 can be recompiled and rerun there completely unchanged, and every `-arch=sm_80` `nvcc` compile command this book has used since Chapter 1 is now launching real kernels on real hardware, not merely compiling them.

**On the Lambda Cloud console (a browser, not a terminal):**

1. Sign up (or sign in) at `cloud.lambda.ai`.
2. Under SSH keys, add a public key -- generate one first if you don't already have one:

```bash
ssh-keygen -t ed25519 -C "dsa-cuda-book"
```

3. Under Instances, launch a new instance: pick any GPU type with availability in a nearby region (a single A10 or A100 is more than enough for every kernel in this book -- none of them need more than one device), select "Don't attach a filesystem" unless you specifically want persistent storage across instances, and choose the SSH key you just added.
4. Wait for the instance's status to change to "Running," then copy the SSH command the console shows you.

**From your own terminal:**

```bash
ssh ubuntu@<the address the console gave you>
```

**Once connected, clone this book's repo and rerun Section A.2's own check:**

```bash
git clone <this book's repository URL>
cd dsa_cuda_from_first_principles
NVDIR=$(dirname $(dirname $(which nvcc)))
nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 209_cuda_runtime_installation_check.cu -o 209_cuda_runtime_installation_check
./209_cuda_runtime_installation_check
```

On Lambda Stack, `nvcc` is already on your `PATH`, so the `NVDIR`/`-I`/`-L` plumbing this book's sandbox needs (because its toolkit was installed via a pip package rather than system-wide) is usually unnecessary there -- a plain `nvcc -arch=sm_80 209_cuda_runtime_installation_check.cu -o 209_cuda_runtime_installation_check` is normally enough. Because this specific walkthrough depends on a real, billed, external machine that this book's own authoring environment has no access to, its exact console output cannot be captured and locked here the way every other piece of output in this book is -- but the three-question diagnosis Section A.2 already built is exactly what will answer, for real, whether it worked.

[COMMON TRAP]
It is tempting to leave a GPU instance running once you are done experimenting, since disconnecting your SSH session does not stop it. Every pay-per-hour GPU provider, Lambda Cloud included, bills for each hour an instance sits in the "Running" state, whether or not you are actively connected to it or a kernel is currently executing -- the only way to stop being billed is to explicitly terminate the instance from the console (or its API), which is worth making the deliberate last step of any real-hardware session, not an afterthought.

## Appendix Summary

Section A.1 confirmed that `nvcc` and `g++` are both present and that this book's bare compile commands (no explicit `-std` flag on the `nvcc` side) already produce C++17, by reading the same facts a version banner shows back out of nvcc's own preprocessor macros. Section A.2 turned Section 2.3's Runtime-API teaching example into a three-question installation checklist -- toolkit, driver, GPU, checked in that dependency order -- that reports honestly whatever is or is not actually present on the machine running it, this book's own driver-less authoring environment included. Section A.3 packaged both of this book's compile-line conventions into a small, genuinely-tested `Makefile`, so that every chapter's own stated compile command is something you run once via `make` rather than retype by hand for every file. Section A.4 pointed past this book's own sandbox entirely: a pay-per-hour Lambda Cloud instance comes with the entire toolchain this appendix just finished verifying already installed, and Section A.2's own checklist is the exact tool for confirming it -- with FOUND replacing NOT FOUND in all three places.
