# ExecutorTorch Tiled Inference

Memory-efficient CPU inference using ExecutorTorch with a multi-model tiled processing approach.

## Overview

This repository demonstrates how to implement **tiled inference** with ExecutorTorch, handling the framework's static shape requirement through a production-ready multi-model pattern. Perfect for edge and mobile deployment where memory is constrained.

## Key Features

- ✅ **100% Pure ExecutorTorch** - No PyTorch runtime fallback needed
- ✅ **Multi-Model Architecture** - Handles static shapes elegantly
- ✅ **Memory Efficient** - Process large inputs in small chunks
- ✅ **Production Ready** - Real-world pattern for edge deployment
- ✅ **Perfect Accuracy** - Results match non-tiled inference exactly

## Quick Start

### Install Dependencies

```bash
pip install executorch torchvision
```

### Run Demo

```bash
python3 hello_executorch_multimodel.py
```

## How It Works

### The Challenge: Static Shapes

ExecutorTorch models have **fixed input shapes** determined at export time. For tiled inference with variable tile sizes, we need a different approach than traditional frameworks.

### The Solution: Multi-Model Pattern

Export separate models for each expected input size:

```python
models = {
    'full': export_model(1024, 1024),    # Full image model
    'tile': export_model(288, 288),       # Tile model (256 + overlap)
}
```

At runtime:
1. Use **full model** for non-tiled inference
2. Use **tile model** for tiled inference (with padding for boundary tiles)

## Architecture

```
┌─────────────────────────────────────┐
│  Export Phase (Offline)             │
├─────────────────────────────────────┤
│ PyTorch Model                       │
│   ↓                                 │
│ Export for each size:               │
│   • Full: 1024×1024 → .pte         │
│   • Tile: 288×288 → .pte           │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  Runtime Phase (On-Device)          │
├─────────────────────────────────────┤
│ Non-tiled: Use full model          │
│                                     │
│ Tiled:                              │
│   1. Load tile model                │
│   2. For each tile:                 │
│      • Extract with overlap         │
│      • Pad to match model size      │
│      • Process with tile model      │
│      • Stitch into output           │
└─────────────────────────────────────┘
```

## Test Results

```
Non-Tiled Inference (ExecutorTorch):
  - Input: 1024×1024 (4 MB)
  - Processing time: 43.35 ms
  
Tiled Inference (ExecutorTorch - 16 tiles):
  - Tile size: 256×256 + 16px overlap
  - Processing time: 87.06 ms
  - Accuracy: Perfect (0.00e+00 difference)
```

## Benefits of Tiled Processing

- ✅ **Reduced Memory Usage** - Process large inputs in manageable chunks
- ✅ **Better Cache Locality** - Smaller working sets fit in CPU cache
- ✅ **Scalable** - Handle inputs larger than available memory
- ✅ **Edge Optimized** - Ideal for memory-constrained devices

## Use Cases

### When to Use Tiled Inference

- ✅ Processing very large images (>2K resolution)
- ✅ Memory-constrained devices (mobile, embedded)  
- ✅ Avoiding Out-Of-Memory errors
- ✅ Batch processing large datasets

### Real-World Applications

- 📱 Mobile image processing apps
- 🤖 Edge AI devices
- 📷 High-resolution image analysis
- 🏥 Medical imaging on embedded systems
- 🛰️ Satellite image processing

## Implementation Details

### File Structure

- `hello_executorch_multimodel.py` - Main demo showcasing multi-model tiled inference
- `.gitignore` - Configured to ignore generated `.pte` model files
- `README.md` - This documentation

### Code Highlights

**Model Export:**
```python
def export_executorch_model(model, h, w, filename):
    exported = torch.export.export(model, (example_input,))
    edge_program = to_edge(exported)
    et_program = edge_program.to_executorch()
    
    with open(filename, "wb") as f:
        f.write(et_program.buffer)
```

**Tiled Inference with Padding:**
```python
# Extract tile
tile = input_data[:, :, start_h:end_h, start_w:end_w].contiguous()

# Pad to expected size for boundary tiles
if tile.shape != expected_shape:
    padded_tile = torch.zeros(expected_shape)
    padded_tile[:, :, :actual_h, :actual_w] = tile
    tile = padded_tile

# Process with ExecutorTorch
result = tile_model.run_method("forward", (tile,))
```

## Environment

- **Python**: 3.10
- **PyTorch**: 2.8.0+cu128
- **ExecutorTorch**: 0.7.0
- **System**: Ubuntu 22.04

## Resources

- [ExecutorTorch Documentation](https://pytorch.org/executorch/)
- [PyTorch Edge](https://pytorch.org/blog/pytorch-edge/)
- [ExecutorTorch GitHub](https://github.com/pytorch/executorch)

## License

MIT License - See individual dependencies for their licenses.

## Contributing

Feel free to open issues or submit PRs for improvements!

---

**Note**: This pattern is used in real mobile/embedded applications where ExecutorTorch's static shapes require pre-compiled models for each expected input configuration.
