# TileLang Hello World - Justfile
# Build automation for the vector addition example

# Default recipe - show help
default:
    @just --list

# Setup: Install uv and create virtual environment with Python 3.10
setup:
    @echo "Setting up environment..."
    @command -v uv >/dev/null 2>&1 || (echo "Installing uv..." && curl -LsSf https://astral.sh/uv/install.sh | sh)
    @echo "✓ uv is available"
    @echo "Installing Python 3.10 (required for TileLang wheels)..."
    @uv python install 3.10
    @echo "Creating virtual environment with Python 3.10..."
    @uv venv --python 3.10
    @echo "✓ Virtual environment created"
    @echo "✓ Setup complete!"

# Install required dependencies using uv
install:
    @echo "Installing TileLang and dependencies with uv..."
    @uv pip install tilelang torch
    @echo "✓ Installation complete!"

# Install using local TileLang from cloth-fit (for macOS ARM)
install-local:
    @echo "Installing TileLang from local source (cloth-fit/thirdparty/tilelang)..."
    @echo "Installing dependencies..."
    @uv pip install torch numpy
    @echo "Installing build tools..."
    @uv pip install setuptools wheel cmake ninja
    @echo "Installing local TileLang in CPU-only mode (this may take a few minutes)..."
    @USE_CUDA=0 USE_ROCM=0 uv pip install -e ../../Developer/cloth-fit/thirdparty/tilelang
    @echo "✓ Installation complete!"

# Sync dependencies from pyproject.toml
sync:
    @echo "Syncing dependencies..."
    @uv sync
    @echo "✓ Sync complete!"

# Run the hello world example
run:
    @echo "Running TileLang Hello World..."
    @echo ""
    @uv run python3 hello_vector_add.py

# Run with CUDA profiling (if available)
run-profile:
    @echo "Running with profiling..."
    @CUDA_VISIBLE_DEVICES=0 uv run python3 hello_vector_add.py

# Clean generated files and cache
clean:
    @echo "Cleaning up..."
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    find . -type f -name "*.pyo" -delete 2>/dev/null || true
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
    @echo "✓ Cleanup complete!"

# Check if dependencies are installed
check:
    @echo "Checking dependencies..."
    @python -c "import tilelang; print('✓ tilelang:', tilelang.__version__)" || echo "✗ tilelang not installed"
    @python -c "import torch; print('✓ torch:', torch.__version__)" || echo "✗ torch not installed"
    @python -c "import torch; print('✓ CUDA available:', torch.cuda.is_available())"

# C++ Build Recipes (using CMake)
# ================================

# Configure CMake build
cpp-configure:
    @echo "Configuring CMake build..."
    @mkdir -p build
    @cd build && cmake .. -DCMAKE_BUILD_TYPE=Release
    @echo "✓ CMake configuration complete"

# Build C++ example
cpp-build: cpp-configure
    @echo "Building C++ example..."
    @cmake --build build --config Release
    @echo "✓ Build complete: ./build/hello_vector_add"

# Run C++ example
cpp-run: cpp-build
    @echo "Running C++ example..."
    @echo ""
    @./build/hello_vector_add

# Clean C++ build artifacts
cpp-clean:
    @echo "Cleaning C++ build artifacts..."
    @rm -rf build
    @echo "✓ C++ build artifacts cleaned"

# Build and run C++ (quick command)
cpp: cpp-build cpp-run

# Show help information
help:
    @echo "TileLang Hello World - Available Commands"
    @echo "=========================================="
    @echo ""
    @echo "Python Commands:"
    @echo "  just install      - Install dependencies (tilelang, torch)"
    @echo "  just run          - Run the Python example"
    @echo "  just run-profile  - Run with CUDA profiling"
    @echo "  just check        - Check installed dependencies"
    @echo ""
    @echo "C++ Commands:"
    @echo "  just cpp-configure - Configure CMake build"
    @echo "  just cpp-build     - Build C++ example"
    @echo "  just cpp-run       - Run C++ example"
    @echo "  just cpp-clean     - Clean C++ build artifacts"
    @echo "  just cpp           - Build and run C++ (quick)"
    @echo ""
    @echo "Utilities:"
    @echo "  just clean        - Clean up generated files"
    @echo "  just help         - Show this help message"
    @echo ""
    @echo "Quick start (Python):"
    @echo "  1. just install"
    @echo "  2. just run"
    @echo ""
    @echo "Quick start (C++):"
    @echo "  1. just cpp"
