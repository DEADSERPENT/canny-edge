#include <iostream>
#include <cstdlib>

// Threshold for deciding between CPU and GPU (in pixels)
// Images smaller than this will use CPU, larger will use GPU
#define CPU_GPU_THRESHOLD (512 * 512)

// Forward declarations for CPU canny implementation
extern "C" {
    void cpu_canny_edge_detection(const char *input_file, const char *output_file);
}

// Hybrid scheduler: decides whether to use CPU or GPU based on image size
bool should_use_gpu(int width, int height)
{
    int totalPixels = width * height;

    std::cout << "Image size: " << width << "x" << height
              << " (" << totalPixels << " pixels)" << std::endl;

    if (totalPixels < CPU_GPU_THRESHOLD) {
        std::cout << "Using CPU for processing (image < "
                  << CPU_GPU_THRESHOLD << " pixels)" << std::endl;
        return false;
    } else {
        std::cout << "Using GPU for processing (image >= "
                  << CPU_GPU_THRESHOLD << " pixels)" << std::endl;
        return true;
    }
}

// Wrapper function to run CPU version
int run_cpu_canny(const char *input_file, const char *output_file)
{
    std::cout << "Launching CPU Canny implementation..." << std::endl;

    // Build the CPU executable path
    std::string cpu_exe = "../canny/canny";
    std::string cmd = cpu_exe + " " + std::string(input_file) + " " + std::string(output_file);

    std::cout << "Executing: " << cmd << std::endl;
    int result = system(cmd.c_str());

    if (result != 0) {
        std::cerr << "CPU Canny failed with exit code: " << result << std::endl;
        return -1;
    }

    std::cout << "CPU Canny completed successfully" << std::endl;
    return 0;
}

// Get image dimensions without fully loading it
bool get_image_dimensions(const char *filename, int *width, int *height)
{
    // This is a simplified version - in production you'd use libpng to peek at header
    // For now, we'll just return true and let the caller handle it
    // The actual dimension check will happen after image is loaded
    return true;
}
