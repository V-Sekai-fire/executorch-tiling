"""
TileLang Hello World: Vector Addition

This simple example demonstrates the core concepts of TileLang by implementing
element-wise vector addition on GPU/CPU.
"""

import tilelang
import tilelang.language as T


@tilelang.jit
def vector_add(N, block_size=256, dtype="float32"):
    """
    Create a vector addition kernel.
    
    Args:
        N: Size of the vectors
        block_size: Number of threads per block
        dtype: Data type for computation
    """
    
    @T.prim_func
    def vector_add_kernel(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
    ):
        # Initialize kernel with grid/block dimensions
        # Each block processes block_size elements
        with T.Kernel(T.ceildiv(N, block_size), threads=block_size) as bx:
            # Allocate thread-local storage for results
            C_local = T.alloc_fragment((block_size,), dtype)
            
            # Parallel computation: each thread handles one element
            for i in T.Parallel(block_size):
                idx = bx * block_size + i
                if idx < N:
                    C_local[i] = A[idx] + B[idx]
            
            # Write results back to global memory
            for i in T.Parallel(block_size):
                idx = bx * block_size + i
                if idx < N:
                    C[idx] = C_local[i]
    
    return vector_add_kernel


def main():
    print("=" * 60)
    print("TileLang Hello World: Vector Addition")
    print("=" * 60)
    
    # Vector size
    N = 1024 * 1024  # 1M elements
    
    print(f"\nVector size: {N:,} elements")
    print(f"Computing: C = A + B (element-wise)")
    
    # Compile the kernel
    print("\n[1/4] Compiling TileLang kernel...")
    kernel = vector_add(N)
    print("✓ Kernel compiled successfully")
    
    # Import torch for testing
    print("\n[2/4] Preparing test data...")
    import torch
    
    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✓ Using device: {device}")
    
    # Create test vectors
    a = torch.randn(N, device=device, dtype=torch.float32)
    b = torch.randn(N, device=device, dtype=torch.float32)
    c = torch.empty(N, device=device, dtype=torch.float32)
    
    # Run the kernel
    print("\n[3/4] Running TileLang kernel...")
    kernel(a, b, c)
    print("✓ Kernel executed")
    
    # Validate results with simple comparison
    print("\n[4/4] Validating results...")
    ref_c = a + b
    
    # Convert to numpy for comparison
    c_np = c.cpu().numpy()
    ref_np = ref_c.cpu().numpy()
    
    # Calculate maximum absolute difference
    max_diff = abs(c_np - ref_np).max()
    
    # Print sample values
    print(f"Sample results (first 5 elements):")
    print(f"  TileLang output: {c_np[:5]}")
    print(f"  Expected (A+B):  {ref_np[:5]}")
    print(f"  Max difference:  {max_diff:.2e}")
    
    # Check if results are correct (tolerance: 1e-5)
    if max_diff < 1e-5:
        print("✓ Results match reference!")
    else:
        print(f"✗ Validation failed: max difference {max_diff:.2e} exceeds tolerance")
        return
    
    # Benchmark performance
    print("\n" + "=" * 60)
    print("Performance Benchmark")
    print("=" * 60)
    
    profiler = kernel.get_profiler(
        tensor_supply_type=tilelang.TensorSupplyType.Normal
    )
    latency = profiler.do_bench()
    
    # Calculate throughput
    bytes_transferred = N * 3 * 4  # 3 vectors * 4 bytes (float32)
    bandwidth_gb_s = (bytes_transferred / 1e9) / (latency / 1000)
    
    print(f"\nLatency: {latency:.3f} ms")
    print(f"Bandwidth: {bandwidth_gb_s:.2f} GB/s")
    print(f"Throughput: {N / (latency * 1e6):.2f} GFLOPS")
    
    print("\n" + "=" * 60)
    print("✓ Hello TileLang - Success!")
    print("=" * 60)


if __name__ == "__main__":
    main()
