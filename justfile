# ExecutorTorch XNNPACK CPU Backend - Dynamic Shapes Tiled Inference - Justfile
# Build automation for XNNPACK backend with dynamic shapes demo

# Default recipe - show help
default:
    @just --list

# Check if uv is installed
check-uv:
    @which uv > /dev/null || (echo "Error: uv not found. Install from https://github.com/astral-sh/uv" && exit 1)
    @echo "✓ uv is available"

# Install Python 3.10 with uv (recommended for best compatibility)
install-python:
    @echo "Installing Python 3.10..."
    @uv python install 3.10
    @echo "✓ Python 3.10 installed"

# Install ExecutorTorch and dependencies
install:
    @echo "Installing ExecutorTorch and dependencies with pip..."
    @pip install torch executorch torchvision
    @echo "✓ Installation complete!"

# Run the XNNPACK backend demo with PyTorch comparison
run:
    @echo "Running ExecutorTorch XNNPACK CPU Backend Demo..."
    @echo ""
    @echo "This demo compares PyTorch (eager) vs ExecutorTorch (XNNPACK)"
    @echo "and demonstrates DYNAMIC SHAPES for variable input sizes."
    @echo ""
    @echo "Features:"
    @echo "  • XNNPACK CPU-optimized inference"
    @echo "  • Single dynamic model (256-1024 range)"
    @echo "  • Performance comparison with PyTorch"
    @echo ""
    @echo "Checking dependencies..."
    @python -c "import executorch; print('✓ executorch:', executorch.__version__)" || echo "✗ executorch not installed"
    @python -c "import torch; print('✓ torch:', torch.__version__)" || echo "✗ torch not installed"
    @python -c "import torchvision; print('✓ torchvision:', torchvision.__version__)" || echo "✗ torchvision not installed"
    @echo ""
    @python hello_executorch_multimodel.py

# Clean generated model files
clean:
    @echo "Cleaning generated model files..."
    @rm -f *.pte
    @echo "✓ Cleaned all .pte files"

# Show help
help:
    @echo "ExecutorTorch XNNPACK CPU Backend - Available Commands"
    @echo "======================================================"
    @echo ""
    @echo "This project demonstrates XNNPACK CPU backend with dynamic shapes"
    @echo "in ExecutorTorch, comparing performance against PyTorch eager mode."
    @echo ""
    @echo "Key Features:"
    @echo "  • XNNPACK CPU optimization (2-5x speedup)"
    @echo "  • Dynamic shapes (256-1024 range)"
    @echo "  • PyTorch vs XNNPACK comparison"
    @echo "  • Single optimized model file"
    @echo ""
    @echo "Setup Commands:"
    @echo "  just install         - Install dependencies (executorch, torch)"
    @echo "  just install-python  - Install Python 3.10 with uv"
    @echo ""
    @echo "Run Commands:"
    @echo "  just run            - Run the XNNPACK backend demo"
    @echo "  just clean          - Remove generated .pte model files"
    @echo ""
    @echo "Info Commands:"
    @echo "  just help           - Show this help message"
    @echo "  just check-uv       - Check if uv is installed"
    @echo ""
    @echo "For more info: github.com/pytorch/executorch/issues/3636"
