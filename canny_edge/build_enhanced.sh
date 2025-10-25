#!/bin/bash

# Build script for enhanced CUDA Canny edge detection
# This script builds the enhanced version with all new features:
# - Adaptive thresholding
# - Hybrid CPU-GPU scheduler
# - Batch processing
# - CUDA streams support
# - Enhanced logging

echo "========================================"
echo "Building Enhanced CUDA Canny"
echo "========================================"

# Default parameters
CUDA_PATH=${CUDA_PATH:-/usr}
TARGET_ARCH=${TARGET_ARCH:-x86_64}
SMS=${SMS:-30}
DEBUG=${DEBUG:-0}

echo "Configuration:"
echo "  CUDA_PATH:    $CUDA_PATH"
echo "  TARGET_ARCH:  $TARGET_ARCH"
echo "  SMS:          $SMS"
echo "  DEBUG:        $DEBUG"
echo ""

# Build the enhanced version
if [ "$DEBUG" = "1" ]; then
    echo "Building in DEBUG mode..."
    make -f Makefile.enhanced enhanced CUDA_PATH=$CUDA_PATH TARGET_ARCH=$TARGET_ARCH SMS=$SMS dbg=1
else
    echo "Building in RELEASE mode..."
    make -f Makefile.enhanced enhanced CUDA_PATH=$CUDA_PATH TARGET_ARCH=$TARGET_ARCH SMS=$SMS
fi

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Build successful!"
    echo "========================================"
    echo "Executable: ./canny_enhanced"
    echo ""
    echo "Features:"
    echo "  - Adaptive thresholding (automatic threshold computation)"
    echo "  - Hybrid CPU/GPU scheduler (size-based selection)"
    echo "  - Batch processing (process entire folders)"
    echo "  - CUDA streams support (parallel pipeline)"
    echo "  - Enhanced performance logging"
    echo ""
    echo "Run with: ./canny_enhanced"
else
    echo ""
    echo "========================================"
    echo "Build failed!"
    echo "========================================"
    exit 1
fi
