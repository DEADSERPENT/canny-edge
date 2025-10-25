#ifndef ENHANCED_LOGGING_H
#define ENHANCED_LOGGING_H

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
    CUDATimer();
    ~CUDATimer();
    void start();
    float stop();
};

// Function declarations
__host__ void print_performance_metrics(const PerformanceMetrics &metrics);
__host__ void log_batch_summary(int numImages, float totalTime, float avgTime);
__host__ void save_metrics_to_csv(const char *filename, const char *imageName,
                                  int width, int height,
                                  const PerformanceMetrics &metrics);

#endif // ENHANCED_LOGGING_H
