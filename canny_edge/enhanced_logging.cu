#include "canny.h"
#include <iostream>
#include <iomanip>

// Structure to hold performance metrics
struct PerformanceMetrics {
    float grayscale_time;
    float blur_time;
    float sobel_time;
    float edge_thin_time;
    float threshold_time;
    float hysteresis_time;
    float total_time;
};

// Enhanced timing using CUDA events
class CUDATimer {
private:
    cudaEvent_t start_event, stop_event;
    bool started;

public:
    CUDATimer() : started(false) {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
    }

    ~CUDATimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    void start() {
        cudaEventRecord(start_event);
        started = true;
    }

    float stop() {
        if (!started) return 0.0f;

        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start_event, stop_event);
        started = false;

        return milliseconds / 1000.0f; // Convert to seconds
    }
};

// Print performance metrics in a formatted table
__host__ void print_performance_metrics(const PerformanceMetrics &metrics)
{
    std::cout << "\n========== Performance Metrics ==========" << std::endl;
    std::cout << std::fixed << std::setprecision(6);

    std::cout << "Grayscale conversion: " << std::setw(10) << metrics.grayscale_time << " s" << std::endl;
    std::cout << "Gaussian blur:        " << std::setw(10) << metrics.blur_time << " s" << std::endl;
    std::cout << "Sobel filter:         " << std::setw(10) << metrics.sobel_time << " s" << std::endl;
    std::cout << "Edge thinning (NMS):  " << std::setw(10) << metrics.edge_thin_time << " s" << std::endl;
    std::cout << "Double thresholding:  " << std::setw(10) << metrics.threshold_time << " s" << std::endl;
    std::cout << "Hysteresis:           " << std::setw(10) << metrics.hysteresis_time << " s" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Total time:           " << std::setw(10) << metrics.total_time << " s" << std::endl;

    // Calculate percentages
    if (metrics.total_time > 0) {
        std::cout << "\n========== Time Distribution ==========" << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Grayscale: " << std::setw(6) << (metrics.grayscale_time / metrics.total_time * 100) << "%" << std::endl;
        std::cout << "Blur:      " << std::setw(6) << (metrics.blur_time / metrics.total_time * 100) << "%" << std::endl;
        std::cout << "Sobel:     " << std::setw(6) << (metrics.sobel_time / metrics.total_time * 100) << "%" << std::endl;
        std::cout << "Edge thin: " << std::setw(6) << (metrics.edge_thin_time / metrics.total_time * 100) << "%" << std::endl;
        std::cout << "Threshold: " << std::setw(6) << (metrics.threshold_time / metrics.total_time * 100) << "%" << std::endl;
        std::cout << "Hysteresis:" << std::setw(6) << (metrics.hysteresis_time / metrics.total_time * 100) << "%" << std::endl;
    }

    std::cout << "========================================\n" << std::endl;
}

// Log batch processing summary
__host__ void log_batch_summary(int numImages, float totalTime, float avgTime)
{
    std::cout << "\n========== Batch Summary ==========" << std::endl;
    std::cout << "Images processed:  " << numImages << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Total time:        " << totalTime << " s" << std::endl;
    std::cout << "Average per image: " << avgTime << " s" << std::endl;
    std::cout << "Throughput:        " << std::fixed << std::setprecision(2)
              << (numImages / totalTime) << " images/s" << std::endl;
    std::cout << "===================================\n" << std::endl;
}

// Save metrics to CSV file for analysis
__host__ void save_metrics_to_csv(const char *filename,
                                  const char *imageName,
                                  int width, int height,
                                  const PerformanceMetrics &metrics)
{
    FILE *fp = fopen(filename, "a");
    if (!fp) {
        std::cerr << "Error: Cannot open " << filename << " for writing" << std::endl;
        return;
    }

    // Write header if file is empty
    fseek(fp, 0, SEEK_END);
    if (ftell(fp) == 0) {
        fprintf(fp, "ImageName,Width,Height,Pixels,Grayscale,Blur,Sobel,EdgeThin,Threshold,Hysteresis,Total\n");
    }

    // Write metrics
    fprintf(fp, "%s,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
            imageName, width, height, width * height,
            metrics.grayscale_time, metrics.blur_time, metrics.sobel_time,
            metrics.edge_thin_time, metrics.threshold_time,
            metrics.hysteresis_time, metrics.total_time);

    fclose(fp);
}
