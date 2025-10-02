# ExecutorTorch Dynamic Shapes Tiled Inference - Justfile
# Build automation for dynamic shapes demo

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

# Run the dynamic shapes demo
run:
    @echo "Running ExecutorTorch Dynamic Shapes Tiled Inference Demo..."
    @echo ""
    @echo "This demo uses a SINGLE dynamic model (256-1024 range)"
    @echo "Based on solution from: github.com/pytorch/executorch/issues/3636"
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
    @echo "ExecutorTorch Dynamic Shapes Tiled Inference - Available Commands"
    @echo "================================================================="
    @echo ""
    @echo "This project demonstrates dynamic shapes in ExecutorTorch, allowing"
    @echo "a single model to handle variable input sizes (256-1024 range)."
    @echo ""
    @echo "Setup Commands:"
    @echo "  just install         - Install dependencies (executorch, torch)"
    @echo "  just install-python  - Install Python 3.10 with uv"
    @echo ""
    @echo "Run Commands:"
    @echo "  just run            - Run the dynamic shapes demo"
    @echo "  just clean          - Remove generated .pte model files"
    @echo ""
    @echo "Info Commands:"
    @echo "  just help           - Show this help message"
    @echo "  just check-uv       - Check if uv is installed"
    @echo ""
    @echo "For more info: github.com/pytorch/executorch/issues/3636"
