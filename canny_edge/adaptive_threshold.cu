#include "canny.h"
#include <algorithm>
#include <cmath>

// Compute histogram of gradient magnitudes
__host__ void compute_histogram(byte *dMagnitude, int h, int w, unsigned int *histogram, int bins)
{
    // Allocate host memory for magnitude data
    byte *hMagnitude = (byte *)malloc(h * w * sizeof(byte));

    // Copy gradient magnitude from device to host
    cudaMemcpy(hMagnitude, dMagnitude, h * w * sizeof(byte), cudaMemcpyDeviceToHost);

    // Initialize histogram
    memset(histogram, 0, bins * sizeof(unsigned int));

    // Compute histogram
    for (int i = 0; i < h * w; i++) {
        int bin = (int)(hMagnitude[i] * bins / 256.0f);
        if (bin >= bins) bin = bins - 1;
        histogram[bin]++;
    }

    free(hMagnitude);
}

// Calculate threshold based on percentile
__host__ float calculate_percentile_threshold(unsigned int *histogram, int bins,
                                               int totalPixels, float percentile)
{
    unsigned int cumulativeSum = 0;
    unsigned int targetCount = (unsigned int)(totalPixels * percentile);

    for (int i = 0; i < bins; i++) {
        cumulativeSum += histogram[i];
        if (cumulativeSum >= targetCount) {
            // Convert bin index back to magnitude value (0-255 range)
            return (i * 255.0f / bins);
        }
    }

    return 255.0f; // fallback
}

// Compute adaptive thresholds based on gradient magnitude histogram
__host__ void compute_adaptive_thresholds(byte *dMagnitude, int h, int w,
                                          float *threshold1, float *threshold2,
                                          float lowPercentile, float highPercentile)
{
    const int BINS = 256;
    unsigned int histogram[BINS];

    // Compute gradient magnitude histogram
    compute_histogram(dMagnitude, h, w, histogram, BINS);

    int totalPixels = h * w;

    // Calculate thresholds based on percentiles
    float t1 = calculate_percentile_threshold(histogram, BINS, totalPixels, lowPercentile);
    float t2 = calculate_percentile_threshold(histogram, BINS, totalPixels, highPercentile);

    // Normalize to 0-1 range
    *threshold1 = t1 / 255.0f;
    *threshold2 = t2 / 255.0f;

    // Ensure t2 >= t1
    if (*threshold2 < *threshold1) {
        float temp = *threshold1;
        *threshold1 = *threshold2;
        *threshold2 = temp;
    }

    // Print computed thresholds for debugging
    std::cout << "Adaptive thresholds computed: t1=" << *threshold1
              << " (raw: " << t1 << "), t2=" << *threshold2
              << " (raw: " << t2 << ")" << std::endl;
}
