#!/bin/bash

# Test script for enhanced CUDA Canny
# This runs with good default parameters

echo "Testing Enhanced CUDA Canny with adaptive thresholding..."
echo ""
echo "Input: ../res/lenna.png"
echo "Output: ../out/lenna_enhanced_adaptive.png"
echo "Settings: Blur=2.0, Adaptive=Yes, Hysteresis=5, Sync=Yes"
echo ""

# Create output directory if needed
mkdir -p ../out

# Run with input piped in
./canny_enhanced <<EOF
1
../res/lenna.png
../out/lenna_enhanced_adaptive.png
2
1
5
1
EOF

echo ""
echo "Done! Check ../out/lenna_enhanced_adaptive.png"
