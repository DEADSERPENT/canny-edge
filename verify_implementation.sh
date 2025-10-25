#!/bin/bash

# Verification script for Enhanced CUDA Canny implementation
# Checks that all required files are present and have non-zero size

echo "================================================"
echo "Enhanced CUDA Canny - Implementation Verification"
echo "================================================"
echo ""

MISSING=0
TOTAL=0

check_file() {
    TOTAL=$((TOTAL + 1))
    if [ -f "$1" ]; then
        SIZE=$(stat -f%z "$1" 2>/dev/null || stat -c%s "$1" 2>/dev/null)
        if [ "$SIZE" -gt 0 ]; then
            printf "✅ %-40s (%6d bytes)\n" "$1" "$SIZE"
        else
            printf "⚠️  %-40s (EMPTY)\n" "$1"
            MISSING=$((MISSING + 1))
        fi
    else
        printf "❌ %-40s (NOT FOUND)\n" "$1"
        MISSING=$((MISSING + 1))
    fi
}

echo "1. Documentation Files"
echo "----------------------"
check_file "README_ENHANCED.md"
check_file "CLAUDE.md"
check_file "QUICKSTART.md"
check_file "IMPLEMENTATION_SUMMARY.md"
echo ""

echo "2. Enhanced CUDA Implementation"
echo "-------------------------------"
check_file "canny_edge/canny_enhanced.cu"
check_file "canny_edge/adaptive_threshold.cu"
check_file "canny_edge/hybrid_scheduler.cpp"
check_file "canny_edge/batch_processor.cpp"
check_file "canny_edge/batch_processor.h"
check_file "canny_edge/parallel_pipeline.cu"
check_file "canny_edge/enhanced_logging.cu"
check_file "canny_edge/enhanced_logging.h"
echo ""

echo "3. Build System"
echo "---------------"
check_file "canny_edge/Makefile.enhanced"
check_file "canny_edge/build_enhanced.sh"
if [ -x "canny_edge/build_enhanced.sh" ]; then
    echo "   ✅ build_enhanced.sh is executable"
else
    echo "   ⚠️  build_enhanced.sh is not executable (run: chmod +x canny_edge/build_enhanced.sh)"
fi
echo ""

echo "4. Original CUDA Files (Modified)"
echo "----------------------------------"
check_file "canny_edge/canny.cu"
check_file "canny_edge/canny.h"
check_file "canny_edge/blur.cu"
check_file "canny_edge/conv2d.cu"
check_file "canny_edge/gray.cu"
check_file "canny_edge/image_prep.cu"
check_file "canny_edge/clock.cu"
echo ""

echo "5. Original CPU Implementation"
echo "------------------------------"
check_file "canny/canny.c"
check_file "canny/blur.c"
check_file "canny/image_prep.c"
check_file "sobel.c"
echo ""

echo "================================================"
echo "Verification Summary"
echo "================================================"
echo "Total files checked: $TOTAL"
echo "Files found: $((TOTAL - MISSING))"
echo "Files missing/empty: $MISSING"
echo ""

if [ $MISSING -eq 0 ]; then
    echo "✅ All files present and ready!"
    echo ""
    echo "Next steps:"
    echo "1. cd canny_edge"
    echo "2. ./build_enhanced.sh"
    echo "3. ./canny_enhanced"
    echo ""
    echo "See QUICKSTART.md for detailed instructions."
    exit 0
else
    echo "⚠️  Some files are missing or empty."
    echo ""
    echo "Please check the missing files listed above."
    exit 1
fi
