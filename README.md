# TileLang Hello World

A simple vector addition example demonstrating the core concepts of TileLang - a domain-specific language for high-performance GPU/CPU tensor kernels.

## What This Does

This project implements element-wise vector addition (`C = A + B`) using TileLang in both **Python** and **C++**, showcasing:

- **Kernel definition** with `@tilelang.jit` decorator
- **Memory management** with shared/local buffers
- **Parallel execution** across GPU threads
- **Performance benchmarking** and validation

## Quick Start

### Prerequisites

- [uv](https://github.com/astral-sh/uv) - Fast Python package installer (will be installed automatically)
- [just](https://github.com/casey/just) - Command runner (optional but recommended)
- CUDA-capable GPU (optional, works on CPU too)
- **Linux or x86_64 architecture** - TileLang currently only provides pre-built wheels for Linux x86_64

**IMPORTANT - Platform Limitation**: TileLang **only supports Linux** (including WSL on Windows). The setup.py explicitly checks for Linux platform and will fail on macOS.

### Using Dev Container (Recommended for macOS/Windows)

This project includes a **VS Code Dev Container** configuration that provides a ready-to-use Linux environment with CUDA support:

1. **Prerequisites:**
   - [Docker Desktop](https://www.docker.com/products/docker-desktop) with GPU support
   - [VS Code](https://code.visualstudio.com/)
   - [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

2. **Open in Dev Container:**
   - Open this folder in VS Code
   - Press `Cmd/Ctrl+Shift+P` and select "Dev Containers: Reopen in Container"
   - Wait for the container to build (first time takes ~5-10 minutes)
   - Once ready, you'll have a full Linux environment with CUDA, Python 3.10, and all tools

3. **Run the example:**
   ```bash
   python3 hello_vector_add.py
   ```

The dev container automatically installs torch, numpy, and tilelang on startup.

### Alternative Options for macOS/Windows Users

1. **Use WSL2** - If on Windows, use Windows Subsystem for Linux 2
2. **Use a Linux VM** - Run Linux in a virtual machine
3. **Remote Linux machine** - SSH into a Linux server

The hello world example in this directory is ready to use once you're on a Linux system.

### Installation

```bash
# Option 1: Automatic setup with uv (recommended)
just setup    # Installs uv and Python 3.11
just install  # Installs tilelang and torch

# Option 2: Using uv directly
uv pip install tilelang torch --system

# Option 3: Sync from pyproject.toml
just sync
```

### Run the Example

```bash
# Using just (recommended)
just run

# Or directly with uv
uv run python3 hello_vector_add.py
```

### Check Dependencies

```bash
just check
```

## What You'll See

```
============================================================
TileLang Hello World: Vector Addition
============================================================

Vector size: 1,048,576 elements
Computing: C = A + B (element-wise)

[1/4] Compiling TileLang kernel...
✓ Kernel compiled successfully

[2/4] Preparing test data...
✓ Using device: cuda

[3/4] Running TileLang kernel...
✓ Kernel executed

[4/4] Validating results...
✓ Results match PyTorch reference!

============================================================
Performance Benchmark
============================================================

Latency: 0.123 ms
Bandwidth: 97.56 GB/s
Throughput: 8.53 GFLOPS

============================================================
✓ Hello TileLang - Success!
============================================================
```

## Code Walkthrough

### The Kernel

```python
@tilelang.jit
def vector_add(N, block_size=256, dtype="float32"):
    @T.prim_func
    def vector_add_kernel(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_size), threads=block_size) as bx:
            C_local = T.alloc_fragment((block_size,), dtype)
            
            for i in T.Parallel(block_size):
                idx = bx * block_size + i
                if idx < N:
                    C_local[i] = A[idx] + B[idx]
            
            for i in T.Parallel(block_size):
                idx = bx * block_size + i
                if idx < N:
                    C[idx] = C_local[i]
    
    return vector_add_kernel
```

**Key Concepts:**

1. `@tilelang.jit` - JIT compilation decorator
2. `T.Kernel()` - Defines grid/block dimensions
3. `T.alloc_fragment()` - Allocates thread-local memory
4. `T.Parallel()` - Parallel loop execution

## Project Structure

```
hello_tilelang/
├── README.md              # This file
├── hello_vector_add.py    # Main vector addition example
└── justfile               # Build automation recipes
```

## C++ Version

This project also includes a **C++ implementation** for testing TileLang via CLI and CMake:

### Quick Start (C++)

```bash
# Build and run C++ example
just cpp

# Or step by step:
just cpp-configure  # Configure CMake
just cpp-build      # Build executable
just cpp-run        # Run the program
```

### Files

- `hello_vector_add.cpp` - C++ implementation (CPU reference)
- `CMakeLists.txt` - CMake build configuration

The C++ version currently uses a CPU reference implementation. TileLang C++ API integration is pending.

## Available Commands

| Command | Description |
|---------|-------------|
| **Python Commands** | |
| `just setup` | Install uv and Python 3.11 |
| `just install` | Install dependencies with uv |
| `just sync` | Sync dependencies from pyproject.toml |
| `just run` | Run the Python example |
| `just run-profile` | Run with CUDA profiling |
| `just check` | Check installed dependencies |
| **C++ Commands** | |
| `just cpp` | Build and run C++ example (quick) |
| `just cpp-configure` | Configure CMake build |
| `just cpp-build` | Build C++ executable |
| `just cpp-run` | Run C++ example |
| `just cpp-clean` | Clean C++ build artifacts |
| **Utilities** | |
| `just clean` | Clean up generated files |
| `just help` | Show help message |

## Next Steps

After mastering this simple example, explore more complex TileLang kernels:

- **Matrix Multiplication (GEMM)** - See `thirdparty/tilelang/examples/gemm/`
- **Flash Attention** - See `thirdparty/tilelang/examples/flash_attention/`
- **Convolution** - See `thirdparty/tilelang/examples/convolution/`

## Learn More

- [TileLang Documentation](https://tilelang.com)
- [TileLang GitHub](https://github.com/tile-ai/tilelang)
- [TileLang Examples](../Developer/cloth-fit/thirdparty/tilelang/examples/)

## License

This example is provided as educational material. TileLang itself is subject to its own license terms.
