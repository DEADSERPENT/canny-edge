# Quick Start Guide - Enhanced CUDA Canny

## Verification: All Files Present ✅

Run this to verify all enhanced files exist:
```bash
cd canny_edge
ls -lh *enhanced* *adaptive* *batch* *hybrid* *parallel* *logging* *.sh Makefile.enhanced
```

You should see:
- `canny_enhanced.cu` (8.4K)
- `adaptive_threshold.cu` (2.6K)
- `batch_processor.cpp` (4.2K)
- `batch_processor.h` (866 bytes)
- `hybrid_scheduler.cpp` (2.1K)
- `parallel_pipeline.cu` (5.2K)
- `enhanced_logging.cu` (4.9K)
- `enhanced_logging.h` (957 bytes)
- `Makefile.enhanced` (5.1K)
- `build_enhanced.sh` (1.8K, executable)

## Step 1: Build

### Option A: Automatic Build (Recommended)
```bash
cd canny_edge
./build_enhanced.sh
```

### Option B: Manual Build with Custom Parameters
```bash
cd canny_edge
make -f Makefile.enhanced enhanced CUDA_PATH=/usr TARGET_ARCH=x86_64 SMS=30
```

Common SMS (compute capability) values:
- `SMS=30` - GT7440, Kepler
- `SMS=35` - K40, K80
- `SMS=50` - Maxwell
- `SMS=52` - GTX 9xx
- `SMS=53` - Tegra TX1/Nano
- `SMS=60` - Pascal GP100
- `SMS=61` - GTX 1080/1070
- `SMS=62` - Tegra TX2
- `SMS=70` - Volta V100
- `SMS=75` - Turing RTX 2080
- `SMS=80` - Ampere A100
- `SMS=86` - RTX 3090

Find your GPU's compute capability:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
```

### Build Output
If successful, you'll see:
```
========================================
Build successful!
========================================
Executable: ./canny_enhanced
```

## Step 2: Prepare Test Data

Create directories:
```bash
cd /home/serpent/cannyedge/canny-edge
mkdir -p res out
```

Copy some test images to `res/`:
```bash
# Example: copy a test image
cp /path/to/your/test.png res/
```

## Step 3: Run Enhanced Version

### Mode 1: Single Image with Adaptive Thresholding

```bash
cd canny_edge
./canny_enhanced
```

Interactive prompts:
```
Select mode (1 or 2): 1
Enter input file (with .png): res/test.png
Enter output file (with .png): out/result.png
Blur stdev: 2
Use adaptive thresholding? (1=yes, 0=no): 1
Hysteresis iters: 5
Sync after each kernel? (1=yes, 0=no): 1
```

### Mode 2: Batch Processing

```bash
cd canny_edge
./canny_enhanced
```

Interactive prompts:
```
Select mode (1 or 2): 2
Enter input directory: res
Enter output directory: out
Blur stdev: 2
Use adaptive thresholding? (1=yes, 0=no): 1
Hysteresis iters: 5
Sync after each kernel? (1=yes, 0=no): 1
```

## Expected Output

### Single Image Output:
```
Image: 1920x1080, Channels: 3
Using GPU for processing (image >= 262144 pixels)
Reading image from file...
Allocating host and device buffers...
Copying image to device...
Converting to grayscale...
Performing canny edge-detection...
Blur filter size: 13
Performing Sobel filter...
Performing edge thinning...
Computing adaptive thresholds...
Adaptive thresholds computed: t1=0.180000 (raw: 45.900000), t2=0.310000 (raw: 79.050000)
Performing double thresholding...
Performing hysteresis...
Convert image back to multi-channel...
Copy image back to host...
Writing image back to file...
Freeing device memory...
Done.
```

### Batch Processing Output:
```
========== Batch Processing Summary ==========
Input directory:  res
Output directory: out
Number of files:  5
Blur stdev:       2.000000
Adaptive mode:    Yes
Hysteresis iters: 5
Sync mode:        Yes
=============================================

[1/5] Processing: image1.png
...
[5/5] Processing: image5.png

==== Batch processing complete! ====
```

## Troubleshooting

### Build Fails: "Cannot find libpng"
```bash
sudo apt-get install libpng-dev  # Ubuntu/Debian
sudo yum install libpng-devel    # CentOS/RHEL
```

### Build Fails: "CUDA_PATH not found"
```bash
# Set CUDA path explicitly
export CUDA_PATH=/usr/local/cuda
./build_enhanced.sh
```

### Build Fails: "Unsupported compute capability"
```bash
# Check your GPU
nvidia-smi

# Get compute capability
nvidia-smi --query-gpu=compute_cap --format=csv,noheader

# Build with correct SMS value (remove decimal point)
# Example: if compute_cap shows 6.1, use SMS=61
make -f Makefile.enhanced enhanced SMS=61
```

### Runtime: "Out of memory"
- Image too large for GPU memory
- Check available memory: `nvidia-smi`
- Hybrid scheduler should automatically use CPU for small images

### Runtime: "No PNG files found"
```bash
# Verify PNG files exist
ls -lh res/*.png

# Check permissions
chmod 644 res/*.png
```

## Verification Tests

### Test 1: Adaptive vs Manual Thresholding
```bash
# Run same image twice with different modes
./canny_enhanced
# Mode 1, adaptive=1, save as out/adaptive.png

./canny_enhanced
# Mode 1, adaptive=0, t1=0.2, t2=0.4, save as out/manual.png

# Compare results visually
```

### Test 2: Hybrid Scheduler
```bash
# Small image (should use CPU)
convert -resize 256x256 res/large.png res/small.png
./canny_enhanced
# Should see: "Using CPU for processing"

# Large image (should use GPU)
./canny_enhanced
# Should see: "Using GPU for processing"
```

### Test 3: Batch Processing
```bash
# Copy multiple images to res/
cp image*.png res/

./canny_enhanced
# Mode 2, process all at once
# Check out/ for results
ls -lh out/
```

## Performance Benchmarking

Run with sync enabled to get timing:
```bash
./canny_enhanced
# Set sync=1
```

Expected output includes:
```
========== Performance Metrics ==========
Grayscale conversion:   0.001240 s
Gaussian blur:          0.004580 s
Sobel filter:           0.002130 s
Edge thinning (NMS):    0.001890 s
Double thresholding:    0.001420 s
Hysteresis:             0.012740 s
----------------------------------------
Total time:             0.024000 s
```

## Next Steps

1. **Test on your images**: Try different types of images
2. **Compare modes**: Test adaptive vs manual thresholding
3. **Benchmark**: Compare enhanced vs original version
4. **Tune parameters**: Adjust blur stdev and hysteresis iterations
5. **Batch process**: Process large image datasets efficiently

## Questions?

Check these files for details:
- `README_ENHANCED.md` - Comprehensive feature documentation
- `CLAUDE.md` - Architecture and development guide
- `Makefile.enhanced` - Build configuration options
- Source files - All well-commented

## Advanced: Custom Modifications

### Change Adaptive Threshold Percentiles
Edit `adaptive_threshold.cu`, line ~47:
```cpp
compute_adaptive_thresholds(dMagnitude, h, w, &threshold1, &threshold2,
                           0.15f,  // Change from 0.1f
                           0.35f); // Change from 0.3f
```

### Adjust CPU/GPU Threshold
Edit `hybrid_scheduler.cpp`, line ~10:
```cpp
#define CPU_GPU_THRESHOLD (512 * 512)  // Change threshold
```

### Enable More CUDA Streams
Edit `parallel_pipeline.cu`, line ~9:
```cpp
#define MAX_STREAMS 4  // Increase to 8 or more
```

Rebuild after any changes:
```bash
make -f Makefile.enhanced clean
./build_enhanced.sh
```
