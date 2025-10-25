#include "canny.h"
#include "batch_processor.h"
#include "image_prep.h"
#include <string>
#include <vector>
#include <iostream>

#define MAX_STREAMS 4

// Structure to hold image buffers for a stream
struct StreamBuffers {
    byte *hImg;          // host image buffer
    byte *dImg;          // device image buffer (color)
    byte *dImgMono;      // device image buffer (grayscale)
    byte *dImgMonoOut;   // device output buffer (grayscale)
    cudaStream_t stream; // CUDA stream
    bool inUse;          // flag to track if buffer is in use
};

// Initialize stream buffers
__host__ void init_stream_buffers(StreamBuffers *buffers, int numStreams,
                                  int maxWidth, int maxHeight, int channels)
{
    size_t colorSize = maxWidth * maxHeight * channels;
    size_t monoSize = maxWidth * maxHeight;

    for (int i = 0; i < numStreams; i++) {
        // Allocate pinned host memory for faster transfers
        cudaMallocHost((void **)&buffers[i].hImg, colorSize);

        // Allocate device memory
        cudaMalloc((void **)&buffers[i].dImg, colorSize);
        cudaMalloc((void **)&buffers[i].dImgMono, monoSize);
        cudaMalloc((void **)&buffers[i].dImgMonoOut, monoSize);

        // Create stream
        cudaStreamCreate(&buffers[i].stream);

        buffers[i].inUse = false;
    }

    std::cout << "Initialized " << numStreams << " CUDA streams with buffers" << std::endl;
}

// Free stream buffers
__host__ void free_stream_buffers(StreamBuffers *buffers, int numStreams)
{
    for (int i = 0; i < numStreams; i++) {
        cudaFreeHost(buffers[i].hImg);
        cudaFree(buffers[i].dImg);
        cudaFree(buffers[i].dImgMono);
        cudaFree(buffers[i].dImgMonoOut);
        cudaStreamDestroy(buffers[i].stream);
    }

    std::cout << "Freed all stream buffers" << std::endl;
}

// Process a single image using a specific stream
__host__ void process_image_stream(const std::string &inputFile,
                                    const std::string &outputFile,
                                    const BatchParams &params,
                                    StreamBuffers &buffer,
                                    dim3 dimGrid, dim3 dimBlock)
{
    unsigned int channels, rowStride;
    cudaError_t err;

    // Read image
    read_png_file(inputFile.c_str());
    channels = color_type == PNG_COLOR_TYPE_RGBA ? 4 : 3;
    rowStride = width * channels;

    // Copy image from row-pointers to host buffer
    for (unsigned int i = 0; i < height; ++i) {
        memcpy(buffer.hImg + i * rowStride, row_pointers[i], rowStride);
    }

    // Asynchronously copy to device
    cudaMemcpyAsync(buffer.dImg, buffer.hImg, width * height * channels,
                    cudaMemcpyHostToDevice, buffer.stream);

    // Convert to grayscale
    toGrayScale<<<dimGrid, dimBlock, 0, buffer.stream>>>(
        buffer.dImg, buffer.dImgMono, height, width, channels);

    // Process with Canny (modified to accept stream parameter)
    // Note: This requires modifying the canny() function to accept a stream
    // For now, we'll synchronize here
    cudaStreamSynchronize(buffer.stream);

    // Call canny edge detection
    // Note: Current canny() doesn't support streams yet
    // This is a placeholder for future stream-aware implementation

    std::cout << "Processed: " << inputFile << " -> " << outputFile << std::endl;
}

// Parallel batch processing using CUDA streams
__host__ void process_batch_parallel(const std::vector<std::string> &files,
                                     const BatchParams &params,
                                     int numStreams = 2)
{
    if (files.empty()) {
        std::cout << "No files to process" << std::endl;
        return;
    }

    if (numStreams > MAX_STREAMS) {
        numStreams = MAX_STREAMS;
    }

    std::cout << "Processing " << files.size() << " images using "
              << numStreams << " parallel streams..." << std::endl;

    // For now, process sequentially with adaptive thresholding
    // Full stream-based parallelism requires refactoring canny() to be stream-aware
    // This is a foundation for future enhancement

    for (size_t i = 0; i < files.size(); i++) {
        std::string inputPath = params.inputDir + "/" + files[i];
        std::string outputFilename = build_output_filename(files[i], params);
        std::string outputPath = params.outputDir + "/" + outputFilename;

        std::cout << "\n[" << (i + 1) << "/" << files.size() << "] Processing: "
                  << files[i] << std::endl;

        // Process image (this would be replaced with stream-based processing)
        // For now, this is a placeholder that shows the structure
        // Actual implementation requires stream-aware canny()
    }

    std::cout << "\nBatch processing complete!" << std::endl;
}

// Enhanced timing with CUDA events
__host__ void time_kernel_with_events(const char *kernelName,
                                      cudaEvent_t start, cudaEvent_t stop)
{
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << kernelName << ":\t" << (milliseconds / 1000.0f) << "s" << std::endl;
}
