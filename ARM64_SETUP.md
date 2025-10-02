# ARM64 / Apple Silicon Setup Notes

## The Issue

TileLang currently only provides pre-built wheels for **Linux x86_64** architecture. If you're running on:
- Apple Silicon (M1/M2/M3) Macs
- ARM64 Linux systems
- Other non-x86_64 architectures

The `pip install tilelang` command will fail with:
```
ERROR: Could not find a version that satisfies the requirement tilelang
```

## The Fix

The dev container configuration has been updated to gracefully handle this:

1. **torch** and **numpy** will install successfully
2. **tilelang** installation will be attempted, but won't block the setup if it fails
3. You'll see a note explaining the architecture limitation

## Options for ARM64 Users

### Option 1: Install from Source (Recommended for Development)

Use the provided justfile command to build TileLang from source:

```bash
just install-local
```

This will:
- Build TileLang from the local source in `../../Developer/cloth-fit/thirdparty/tilelang`
- Use CPU-only mode (no CUDA)
- Take a few minutes to compile

### Option 2: Use x86_64 Emulation

Run the dev container with x86_64 emulation:

1. In your Docker Desktop settings, enable Rosetta emulation (macOS only)
2. Or use `--platform linux/amd64` when building the container

### Option 3: Use a Remote x86_64 Machine

If you need GPU acceleration or faster builds:
- Use a cloud instance (AWS, GCP, Azure)
- SSH into a Linux x86_64 server
- Use GitHub Codespaces with x86_64

## Current Setup Status

After the dev container builds, you'll have:
- ✅ Python 3.10 virtual environment
- ✅ PyTorch (CPU version for ARM64)
- ✅ NumPy
- ⚠️  TileLang (may require source installation on ARM64)

## Next Steps

1. If you're just exploring the project structure: **no action needed**
2. If you want to run the examples: Use `just install-local` to build from source
3. If you need GPU support: Use an x86_64 Linux system

## Questions?

See the main [README.md](README.md) for more information about TileLang's platform requirements.
