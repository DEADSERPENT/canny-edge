#include "canny.h"
#include "batch_processor.h"
#include "image_prep.h"
#include <string>
#include <iostream>
#include <vector>

// Global variables (from canny.cu)
extern dim3 dimGrid, dimBlock;
extern bool doSync;
extern cudaError_t err;

// Forward declarations
extern __host__ void canny(byte *dImg, byte *dImgOut,
                          float blurStd, float threshold1, float threshold2,
                          int hystIters, bool useAdaptive);

extern __host__ bool should_use_gpu(int width, int height);
extern __host__ int run_cpu_canny(const char *input_file, const char *output_file);

// Process a single image with all enhancements
__host__ int process_single_image(const std::string &inputFile,
                                  const std::string &outputFile,
                                  float blurStd, bool useAdaptive,
                                  float threshold1, float threshold2,
                                  unsigned int hystIters, bool sync)
{
    unsigned int channels, rowStride;
    byte *hImg, *dImg, *dImgMono, *dImgMonoOut;
    clock_t *tGray, *tOverall;

    // Read image
    std::cout << "Reading image from file..." << std::endl;
    read_png_file(inputFile.c_str());
    channels = color_type == PNG_COLOR_TYPE_RGBA ? 4 : 3;
    rowStride = width * channels;

    std::cout << "Image: " << width << "x" << height
              << ", Channels: " << channels << std::endl;

    // Check if we should use GPU or CPU
    if (!should_use_gpu(width, height)) {
        return run_cpu_canny(inputFile.c_str(), outputFile.c_str());
    }

    // Allocate memory
    std::cout << "Allocating host and device buffers..." << std::endl;
    hImg = (byte *)malloc(width * height * channels);
    CUDAERR(cudaMalloc((void **)&dImg, width * height * channels),
            "cudaMalloc dImg");
    CUDAERR(cudaMalloc((void **)&dImgMono, width * height),
            "cudaMalloc dImgMono");
    CUDAERR(cudaMalloc((void **)&dImgMonoOut, width * height),
            "cudaMalloc dImgMonoOut");

    // Copy image from row-pointers to host buffer
    for (unsigned int i = 0; i < height; ++i) {
        memcpy(hImg + i * rowStride, row_pointers[i], rowStride);
    }

    // Copy image to device
    std::cout << "Copying image to device..." << std::endl;
    CUDAERR(cudaMemcpy(dImg, hImg, width * height * channels,
                      cudaMemcpyHostToDevice), "cudaMemcpy to device");

    // Set kernel parameters
    dimGrid = dim3(ceil(rowStride * 1. / bs), ceil(height * 1. / bs), 1);
    dimBlock = dim3(bs, bs, 1);

    // Convert to grayscale
    std::cout << "Converting to grayscale..." << std::endl;
    toGrayScale<<<dimGrid, dimBlock>>>(dImg, dImgMono, height, width, channels);
    CUDAERR(cudaGetLastError(), "launch toGrayScale kernel");
    if (sync) {
        cudaDeviceSynchronize();
    }

    // Canny edge detection with adaptive thresholding
    std::cout << "Performing canny edge-detection..." << std::endl;
    doSync = sync;
    canny(dImgMono, dImgMonoOut, blurStd, threshold1, threshold2,
          hystIters, useAdaptive);

    // Convert back from grayscale
    std::cout << "Convert image back to multi-channel..." << std::endl;
    fromGrayScale<<<dimGrid, dimBlock>>>(dImgMonoOut, dImg,
                                        height, width, channels);
    CUDAERR(cudaGetLastError(), "launch fromGrayScale kernel");
    cudaDeviceSynchronize();

    // Copy image back to host
    std::cout << "Copy image back to host..." << std::endl;
    CUDAERR(cudaMemcpy(hImg, dImg, width * height * channels,
                      cudaMemcpyDeviceToHost), "cudaMemcpy to host");

    // Copy back to row_pointers
    for (unsigned int i = 0; i < height; ++i) {
        memcpy(row_pointers[i], hImg + i * rowStride, rowStride);
    }

    // Write output image
    std::cout << "Writing image back to file..." << std::endl;
    write_png_file(outputFile.c_str());

    // Cleanup
    std::cout << "Freeing device memory..." << std::endl;
    free(hImg);
    CUDAERR(cudaFree(dImg), "freeing dImg");
    CUDAERR(cudaFree(dImgMono), "freeing dImgMono");
    CUDAERR(cudaFree(dImgMonoOut), "freeing dImgMonoOut");

    std::cout << "Done." << std::endl;
    return 0;
}

// Main function with batch processing support
__host__ int main(int argc, char **argv)
{
    int mode;
    bool useAdaptive;
    float blurStd, threshold1, threshold2;
    unsigned int hystIters;
    bool sync;

    std::cout << "==== Enhanced CUDA Canny Edge Detection ====" << std::endl;
    std::cout << "1. Single image mode" << std::endl;
    std::cout << "2. Batch processing mode" << std::endl;
    std::cout << "Select mode (1 or 2): ";
    std::cin >> mode;

    if (mode == 1) {
        // Single image mode
        std::string inFile, outFile;

        std::cout << "\n--- Single Image Mode ---" << std::endl;
        std::cout << "Enter input file (with .png): ";
        std::cin >> inFile;

        std::cout << "Enter output file (with .png): ";
        std::cin >> outFile;

        std::cout << "Blur stdev: ";
        std::cin >> blurStd;

        std::cout << "Use adaptive thresholding? (1=yes, 0=no): ";
        std::cin >> useAdaptive;

        if (!useAdaptive) {
            std::cout << "Threshold 1: ";
            std::cin >> threshold1;

            std::cout << "Threshold 2: ";
            std::cin >> threshold2;
        } else {
            threshold1 = 0.0f;
            threshold2 = 0.0f;
        }

        std::cout << "Hysteresis iters: ";
        std::cin >> hystIters;

        std::cout << "Sync after each kernel? (1=yes, 0=no): ";
        std::cin >> sync;

        return process_single_image(inFile, outFile, blurStd, useAdaptive,
                                   threshold1, threshold2, hystIters, sync);

    } else if (mode == 2) {
        // Batch processing mode
        BatchParams params;

        std::cout << "\n--- Batch Processing Mode ---" << std::endl;
        std::cout << "Enter input directory: ";
        std::cin >> params.inputDir;

        std::cout << "Enter output directory: ";
        std::cin >> params.outputDir;

        // Ensure output directory exists
        if (!ensure_directory_exists(params.outputDir)) {
            return -1;
        }

        // Get list of PNG files
        std::vector<std::string> files = get_png_files(params.inputDir);
        if (files.empty()) {
            std::cout << "No PNG files found in " << params.inputDir << std::endl;
            return -1;
        }

        std::cout << "Blur stdev: ";
        std::cin >> params.blurStd;

        std::cout << "Use adaptive thresholding? (1=yes, 0=no): ";
        std::cin >> params.useAdaptive;

        if (!params.useAdaptive) {
            std::cout << "Threshold 1: ";
            std::cin >> params.threshold1;

            std::cout << "Threshold 2: ";
            std::cin >> params.threshold2;
        } else {
            params.threshold1 = 0.0f;
            params.threshold2 = 0.0f;
        }

        std::cout << "Hysteresis iters: ";
        std::cin >> params.hystIters;

        std::cout << "Sync after each kernel? (1=yes, 0=no): ";
        std::cin >> params.doSync;

        // Print summary
        print_batch_summary(files, params);

        // Process all files
        for (size_t i = 0; i < files.size(); i++) {
            std::string inputPath = params.inputDir + "/" + files[i];
            std::string outputFilename = build_output_filename(files[i], params);
            std::string outputPath = params.outputDir + "/" + outputFilename;

            std::cout << "\n[" << (i + 1) << "/" << files.size() << "] "
                      << "Processing: " << files[i] << std::endl;

            int result = process_single_image(inputPath, outputPath,
                                             params.blurStd, params.useAdaptive,
                                             params.threshold1, params.threshold2,
                                             params.hystIters, params.doSync);

            if (result != 0) {
                std::cerr << "Failed to process " << files[i] << std::endl;
            }
        }

        std::cout << "\n==== Batch processing complete! ====" << std::endl;
        return 0;

    } else {
        std::cerr << "Invalid mode selected" << std::endl;
        return -1;
    }
}
