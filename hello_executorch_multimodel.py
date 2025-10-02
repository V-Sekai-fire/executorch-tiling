"""ExecutorTorch Dynamic Shapes Tiled Inference Demo.

Demonstrates ExecutorTorch tiled inference using dynamic shapes to handle
variable input sizes with a single model, eliminating the need for multiple
static models.

This implementation uses torch.export.Dim to define dynamic dimensions and
ConstraintBasedSymShapeEvalPass for proper memory planning, allowing a single
.pte file to handle any input size within the specified min/max range.

Based on solution from: https://github.com/pytorch/executorch/issues/3636
"""

import torch
import torch.nn as nn
import time
import tracemalloc
from typing import Tuple
from torch.export import Dim
from executorch.exir import to_edge, ExecutorchBackendConfig
from executorch.exir.passes.sym_shape_eval_pass import ConstraintBasedSymShapeEvalPass
from executorch.extension.pybindings.portable_lib import _load_for_executorch


class SimpleEdgeDetector(nn.Module):
    """Simple edge detection model using 3x3 convolution.
    
    Implements a basic edge detection filter using a fixed Laplacian-style
    kernel. The output is clamped to [0, 1] range.
    
    Attributes:
        conv: Conv2d layer with fixed edge detection kernel weights
    """
    
    def __init__(self):
        """Initialize the edge detector with a Laplacian kernel."""
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        
        edge_kernel = torch.tensor([[
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ]], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.conv.weight = nn.Parameter(edge_kernel)
        self.conv.weight.requires_grad = False
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply edge detection to input tensor.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
            
        Returns:
            Edge-detected tensor clamped to [0, 1] range
        """
        return torch.clamp(self.conv(x), 0, 1)


def export_executorch_model_dynamic(
    model: nn.Module, 
    example_h: int,
    example_w: int,
    min_size: int,
    max_size: int,
    filename: str
) -> str:
    """Export ExecutorTorch model with dynamic height and width dimensions.
    
    Creates a single .pte model file that can handle variable input sizes within
    the specified min/max range. Uses torch.export.Dim to define dynamic dimensions
    and ConstraintBasedSymShapeEvalPass for proper memory planning based on maximum
    constraints rather than example input size.
    
    The export process follows three steps:
    1. Export to core ATen with dynamic shapes specification
    2. Convert to ExecutorTorch Edge dialect
    3. Generate ExecutorTorch program with dynamic shape support
    
    Args:
        model: PyTorch model to export
        example_h: Example height for export tracing
        example_w: Example width for export tracing
        min_size: Minimum dimension size for both height and width
        max_size: Maximum dimension size for both height and width
        filename: Output .pte filename
    
    Returns:
        Path to the exported model file
        
    Raises:
        Exception: If export process fails at any step
    """
    print(f"\n{'='*60}")
    print(f"Exporting Dynamic ExecutorTorch Model")
    print(f"{'='*60}")
    print(f"Example input: {example_h}x{example_w}")
    print(f"Dynamic range: {min_size} to {max_size}")
    
    model.eval()
    example_input = torch.randn(1, 1, example_h, example_w)
    
    try:
        dynamic_h = Dim("height", min=min_size, max=max_size)
        dynamic_w = Dim("width", min=min_size, max=max_size)
        dynamic_shapes = {"x": {2: dynamic_h, 3: dynamic_w}}
        
        print(f"\nDynamic shapes configuration:")
        print(f"  Height dimension (axis 2): {min_size} to {max_size}")
        print(f"  Width dimension (axis 3): {min_size} to {max_size}")
        
        print(f"\n[1/3] Exporting to core ATen with dynamic shapes...")
        exported = torch.export.export(
            model,
            (example_input,),
            dynamic_shapes=dynamic_shapes
        )
        print(f"✓ Core ATen export complete")
        print(f"  Graph signature: {exported.graph_signature}")
        
        print(f"\n[2/3] Converting to Edge dialect...")
        edge_program = to_edge(exported)
        print(f"✓ Edge program created")
        
        print(f"\n[3/3] Exporting to ExecutorTorch...")
        et_program = edge_program.to_executorch(
            config=ExecutorchBackendConfig(
                sym_shape_eval_pass=ConstraintBasedSymShapeEvalPass(),
            )
        )
        print(f"✓ ExecutorTorch program created")
        
        with open(filename, "wb") as f:
            f.write(et_program.buffer)
        
        print(f"\n✓ Model exported successfully to: {filename}")
        return filename
        
    except Exception as e:
        print(f"\n✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def load_executorch_model(filename: str):
    """Load ExecutorTorch model from .pte file.
    
    Args:
        filename: Path to the .pte model file
        
    Returns:
        Loaded ExecutorTorch runtime instance
        
    Raises:
        Exception: If model loading fails
    """
    print(f"\n{'='*60}")
    print(f"Loading ExecutorTorch Model")
    print(f"{'='*60}")
    
    try:
        runtime = _load_for_executorch(filename)
        print(f"✓ Loaded: {filename}")
        return runtime
    except Exception as e:
        print(f"✗ Failed to load {filename}: {e}")
        raise


def executorch_inference(
    model,
    input_data: torch.Tensor,
    description: str
) -> Tuple[torch.Tensor, float, int]:
    """Run inference with ExecutorTorch dynamic model.
    
    Executes the forward method on the loaded ExecutorTorch runtime and measures
    execution time and peak memory usage. The dynamic model automatically handles
    any input size within its configured range.
    
    Args:
        model: Loaded ExecutorTorch runtime instance
        input_data: Input tensor of shape (batch, channels, height, width)
        description: Human-readable description for logging
        
    Returns:
        Tuple containing:
            - output: Model output tensor
            - elapsed: Execution time in seconds
            - peak: Peak memory usage in bytes
    """
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    
    tracemalloc.start()
    start_time = time.time()
    
    result = model.run_method("forward", (input_data,))
    output = result[0]
    
    elapsed = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"Input shape: {input_data.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Processing time: {elapsed*1000:.2f} ms")
    print(f"Peak memory: {peak / 1024**2:.2f} MB")
    
    return output, elapsed, peak


def executorch_tiled_inference(
    model,
    input_data: torch.Tensor,
    tile_size: int = 256,
    overlap: int = 16
) -> Tuple[torch.Tensor, float, int]:
    """Process image using tiled inference with ExecutorTorch dynamic model.
    
    Divides the input image into overlapping tiles, processes each tile with the
    dynamic model, and stitches results back together. The dynamic model automatically
    handles variable tile sizes at boundaries without requiring padding, eliminating
    the need for multiple static models.
    
    The tiling strategy includes overlap to avoid edge artifacts. For interior tiles,
    the overlap region is discarded during stitching. Boundary tiles may have different
    dimensions than interior tiles, which the dynamic model handles seamlessly.
    
    Args:
        model: Loaded ExecutorTorch runtime instance with dynamic shape support
        input_data: Input tensor of shape (batch, channels, height, width)
        tile_size: Base tile size in pixels (default: 256)
        overlap: Overlap width in pixels on each side (default: 16)
        
    Returns:
        Tuple containing:
            - output: Stitched output tensor matching input dimensions
            - elapsed: Total processing time in seconds
            - peak: Peak memory usage in bytes
    """
    print(f"\n{'='*60}")
    print(f"Tiled Inference (Dynamic ExecutorTorch Model)")
    print(f"{'='*60}")
    
    tracemalloc.start()
    start_time = time.time()
    
    b, c, h, w = input_data.shape
    output = torch.zeros_like(input_data)
    
    num_tiles_h = (h + tile_size - 1) // tile_size
    num_tiles_w = (w + tile_size - 1) // tile_size
    
    print(f"Input shape: {input_data.shape}")
    print(f"Tile size: {tile_size}x{tile_size} (overlap: {overlap}px)")
    print(f"Number of tiles: {num_tiles_h} x {num_tiles_w} = {num_tiles_h * num_tiles_w}")
    
    tiles_processed = 0
    tile_sizes_used = set()
    
    for i in range(num_tiles_h):
        for j in range(num_tiles_w):
            start_h = max(0, i * tile_size - overlap)
            end_h = min(h, (i + 1) * tile_size + overlap)
            start_w = max(0, j * tile_size - overlap)
            end_w = min(w, (j + 1) * tile_size + overlap)
            
            tile = input_data[:, :, start_h:end_h, start_w:end_w].contiguous()
            actual_h, actual_w = tile.shape[2], tile.shape[3]
            tile_sizes_used.add((actual_h, actual_w))
            
            result = model.run_method("forward", (tile,))
            tile_output = result[0]
            
            valid_start_h = overlap if i > 0 else 0
            valid_end_h = valid_start_h + min(tile_size, h - i * tile_size)
            valid_start_w = overlap if j > 0 else 0
            valid_end_w = valid_start_w + min(tile_size, w - j * tile_size)
            
            out_start_h = i * tile_size
            out_end_h = min(h, (i + 1) * tile_size)
            out_start_w = j * tile_size
            out_end_w = min(w, (j + 1) * tile_size)
            
            valid_h = out_end_h - out_start_h
            valid_w = out_end_w - out_start_w
            output[:, :, out_start_h:out_end_h, out_start_w:out_end_w] = \
                tile_output[:, :, valid_start_h:valid_start_h+valid_h, valid_start_w:valid_start_w+valid_w]
            
            tiles_processed += 1
    
    elapsed = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"Tiles processed: {tiles_processed}")
    print(f"Unique tile sizes handled: {len(tile_sizes_used)}")
    print(f"  Sizes: {sorted(tile_sizes_used)}")
    print(f"Processing time: {elapsed*1000:.2f} ms")
    print(f"Peak memory: {peak / 1024**2:.2f} MB")
    
    return output, elapsed, peak


def main():
    """Demonstrate ExecutorTorch dynamic shapes with tiled inference.
    
    This function demonstrates the complete workflow of using dynamic shapes in
    ExecutorTorch for tiled image processing:
    
    1. Creates and exports a single dynamic model (256-1024 range)
    2. Loads the dynamic model runtime
    3. Tests full-image inference at 1024x1024
    4. Tests tiled inference with overlap handling
    5. Tests flexibility with different input size (512x512)
    6. Validates and compares results
    
    The demonstration shows how a single dynamic model eliminates the need for
    multiple static models while handling variable tile sizes automatically.
    """
    print("="*70)
    print(" " * 10 + "ExecutorTorch Dynamic Shapes Tiled Inference Demo")
    print("="*70)
    print("\nThis demo uses DYNAMIC SHAPES to handle variable input sizes")
    print("with a SINGLE ExecutorTorch model (no multiple static models needed).")
    
    print("\n[Step 1] Creating edge detection model...")
    model = SimpleEdgeDetector()
    print("✓ Model created")
    
    tile_size = 256
    overlap = 16
    min_size = tile_size
    max_size = 1024
    
    print(f"\n[Step 2] Exporting dynamic model...")
    print(f"  This single model will handle ALL sizes from {min_size}x{min_size} to {max_size}x{max_size}")
    
    model_file = export_executorch_model_dynamic(
        model,
        example_h=tile_size + 2 * overlap,
        example_w=tile_size + 2 * overlap,
        min_size=min_size,
        max_size=max_size,
        filename="model_dynamic.pte"
    )
    
    print(f"\n[Step 3] Loading dynamic model...")
    et_model = load_executorch_model(model_file)
    
    print(f"\n[Step 4] Creating test input (1024x1024 image)...")
    input_data = torch.randn(1, 1, 1024, 1024)
    print(f"✓ Input created: {input_data.shape}")
    print(f"  Input size: {input_data.numel() * 4 / 1024**2:.2f} MB")
    
    output_full, time_full, mem_full = executorch_inference(
        et_model,
        input_data,
        "Full Image Inference (Dynamic Model @ 1024x1024)"
    )
    
    output_tiled, time_tiled, mem_tiled = executorch_tiled_inference(
        et_model,
        input_data,
        tile_size=tile_size,
        overlap=overlap
    )
    
    print(f"\n[Step 5] Testing dynamic model with different input size...")
    test_input = torch.randn(1, 1, 512, 512)
    output_512, time_512, mem_512 = executorch_inference(
        et_model,
        test_input,
        "Dynamic Model @ 512x512 (Same Model!)"
    )
    
    print(f"\n{'='*60}")
    print("Validation & Comparison")
    print(f"{'='*60}")
    
    max_diff = torch.abs(output_full - output_tiled).max().item()
    print(f"Max difference (full vs tiled): {max_diff:.2e}")
    if max_diff < 1e-5:
        print("✓ Results match perfectly!")
    elif max_diff < 1e-3:
        print("✓ Results match (minor floating point differences)")
    else:
        print("⚠️ Results differ - check implementation")
    
    print(f"\nPerformance Comparison:")
    print(f"  Full image time:  {time_full*1000:.2f} ms")
    print(f"  Tiled time:       {time_tiled*1000:.2f} ms")
    print(f"  512x512 time:     {time_512*1000:.2f} ms")
    print(f"  Tiling overhead:  {((time_tiled/time_full - 1) * 100):.1f}%")
    
    print(f"\nMemory Comparison:")
    print(f"  Full image peak:  {mem_full / 1024**2:.2f} MB")
    print(f"  Tiled peak:       {mem_tiled / 1024**2:.2f} MB")
    print(f"  512x512 peak:     {mem_512 / 1024**2:.2f} MB")
    if mem_tiled > 0 and mem_full > 0:
        memory_savings = (1 - mem_tiled / mem_full) * 100
        print(f"  Memory savings (tiled): {memory_savings:.1f}%")
    
    print(f"\n{'='*70}")
    print("Key Achievements:")
    print(f"{'='*70}")
    print("✓ Single dynamic model handles all input sizes (256-1024)")
    print("✓ No padding required for different tile sizes")
    print("✓ No model selection logic needed")
    print("✓ Smaller deployment footprint (1 model vs multiple)")
    print("✓ Memory-efficient tiled processing")
    print("✓ Production-ready for edge deployment")
    print(f"\n{'='*70}")
    print("Dynamic Shapes Benefits:")
    print(f"{'='*70}")
    print("• One model file instead of multiple static models")
    print("• Flexible input dimensions within defined range")
    print("• Automatic handling of boundary tiles (no padding)")
    print("• Simplified deployment and maintenance")
    print("• Same approach works for Android/iOS/embedded")
    print(f"\nBased on solution from: github.com/pytorch/executorch/issues/3636")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
