# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements Canny edge detection in two versions:
- **CUDA version** (`canny_edge/`): GPU-accelerated implementation using NVIDIA CUDA
- **CPU version** (`canny/`): Pure C implementation using libpng

Based on the paper "Canny edge detection on NVIDIA CUDA" (https://ieeexplore.ieee.org/abstract/document/4563088)

## Architecture

### Canny Edge Detection Pipeline

Both implementations follow the standard Canny edge detection algorithm:

1. **Grayscale conversion** - Convert RGB image to single-channel grayscale
2. **Gaussian blur** - Apply Gaussian filter to reduce noise (uses separable or 2D convolution)
3. **Sobel filter** - Calculate gradient magnitude and direction using Sobel operators
4. **Edge thinning** (Non-maximum suppression) - Thin edges to single-pixel width based on gradient direction
5. **Double thresholding** - Classify pixels as strong edges, weak edges, or non-edges using two thresholds
6. **Hysteresis** - Connect weak edges to strong edges through iterative propagation

### Code Organization

**CUDA version** (`canny_edge/`):
- `canny.cu` - Main entry point and pipeline orchestration
- `blur.cu` - Gaussian blur kernels (both separable and 2D convolution)
- `conv2d.cu` - 2D convolution kernel
- `gray.cu` - Grayscale conversion kernel
- `image_prep.cu` - PNG image I/O using libpng
- `clock.cu` - Performance timing utilities
- `canny.h` - Common definitions and kernel declarations

**CPU version** (`canny/`):
- `canny.c` - Main entry point and CPU implementation of edge thinning, hysteresis
- `blur.c` - Gaussian blur CPU implementation
- `image_prep.c` - PNG image I/O
- `sobel.c` (root) - Shared Sobel filter implementation used by CPU version

**Shared code**:
- `sobel.c` in root directory contains Sobel filter implementations shared between versions

## Build Commands

### CUDA Version

Build with specific architecture (recommended approach):
```bash
make -C canny_edge CUDA_PATH=/usr TARGET_ARCH=x86_64 SMS=30
```

Or use the convenience script:
```bash
cd canny_edge && ./buildx86.sh
```

**Important Makefile variables:**
- `CUDA_PATH` - Path to CUDA installation (default: `/usr/local/cuda-10.0`)
- `TARGET_ARCH` - Target architecture (`x86_64` or `aarch64`)
- `SMS` - Compute capability (30 for GT7440, 53 for Jetson TX1/Nano, 62 for TX2)
- `dbg=1` - Enable debug build with `-g -G` flags

Clean:
```bash
make -C canny_edge clean
```

### CPU Version

Build:
```bash
make -C canny
```

Clean:
```bash
make -C canny clean
```

**Note:** CPU version requires libpng development headers.

## Running the Applications

### CUDA Version

The executable is `canny_edge/canny`. It prompts for parameters interactively:

```bash
canny_edge/canny
```

Example prompts:
- **Input file:** `res/lizard` (without .png extension)
- **Output file:** `out/lizard` (without .png extension)
- **Blur stdev:** `2` (controls Gaussian blur strength)
- **Threshold 1:** `0.2` (lower threshold for weak edges, 0-1 range)
- **Threshold 2:** `0.4` (upper threshold for strong edges, 0-1 range)
- **Hysteresis iterations:** `5` (number of edge propagation iterations)
- **Sync after each kernel:** `1` (1=sync for timing, 0=async for max performance)

Output filename format: `[OUTFILE]_bs[BLURSIZE]_th[THRESHOLD1]_th[THRESHOLD2].png`

### CPU Version

The executable is `canny/canny`. It takes command-line arguments:

```bash
canny/canny <input.png> <output.png>
```

Example:
```bash
canny/canny res/lizard.png out/result.png
```

## Development Notes

### CUDA Implementation Details

- **Kernel grid/block dimensions:** Set globally in `dimGrid` and `dimBlock` variables
- **Shared memory versions:** Some kernels have `_shm` variants (e.g., `sobel_shm`, `hysteresis_shm`) for optimized shared memory usage
- **Separable filters:** The blur can use separable convolution (`blur_sep`) for better performance
- **Synchronization:** The `doSync` flag controls whether to sync after each kernel for accurate timing
- **Constant memory:** Filter coefficients can be stored in constant memory for faster access

### Image Requirements

- Both versions expect PNG images
- Input images should be in an accessible directory (e.g., `res/`)
- Output directory must exist before running (e.g., `out/`)

### Debugging

For CUDA version, build with `dbg=1`:
```bash
make -C canny_edge CUDA_PATH=/usr TARGET_ARCH=x86_64 SMS=30 dbg=1
```

VS Code launch configuration exists in `.vscode/launch.json` (currently configured for macOS paths).

### Performance Considerations

- **CUDA version:** Synchronization affects timing measurements but not overall performance when disabled
- **Hysteresis iterations:** More iterations = better edge connectivity but slower performance
- **Blur size:** Larger blur standard deviation = larger filter kernel = slower but smoother
- **Shared memory kernels:** Generally faster than global memory versions for small tile sizes
