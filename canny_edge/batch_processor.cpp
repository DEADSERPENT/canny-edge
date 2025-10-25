#include <iostream>
#include <string>
#include <vector>
#include <dirent.h>
#include <sys/stat.h>
#include <cstring>

// Structure to hold batch processing parameters
struct BatchParams {
    std::string inputDir;
    std::string outputDir;
    float blurStd;
    bool useAdaptive;
    float threshold1;
    float threshold2;
    unsigned int hystIters;
    bool doSync;
};

// Check if a file has .png extension
bool is_png_file(const std::string &filename)
{
    if (filename.length() < 4) return false;

    std::string ext = filename.substr(filename.length() - 4);
    return (ext == ".png" || ext == ".PNG");
}

// Get list of PNG files in a directory
std::vector<std::string> get_png_files(const std::string &directory)
{
    std::vector<std::string> files;
    DIR *dir = opendir(directory.c_str());

    if (dir == NULL) {
        std::cerr << "Error: Cannot open directory " << directory << std::endl;
        return files;
    }

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        std::string filename = entry->d_name;

        // Skip . and ..
        if (filename == "." || filename == "..") continue;

        // Check if it's a PNG file
        if (is_png_file(filename)) {
            files.push_back(filename);
        }
    }

    closedir(dir);

    std::cout << "Found " << files.size() << " PNG files in " << directory << std::endl;
    return files;
}

// Create directory if it doesn't exist
bool ensure_directory_exists(const std::string &directory)
{
    struct stat st;

    if (stat(directory.c_str(), &st) == 0) {
        if (S_ISDIR(st.st_mode)) {
            return true; // Directory exists
        } else {
            std::cerr << "Error: " << directory << " exists but is not a directory" << std::endl;
            return false;
        }
    }

    // Directory doesn't exist, create it
    #ifdef _WIN32
        if (mkdir(directory.c_str()) == 0) {
    #else
        if (mkdir(directory.c_str(), 0755) == 0) {
    #endif
            std::cout << "Created output directory: " << directory << std::endl;
            return true;
        } else {
            std::cerr << "Error: Failed to create directory " << directory << std::endl;
            return false;
        }
}

// Extract filename without extension
std::string get_filename_without_extension(const std::string &filename)
{
    size_t lastdot = filename.find_last_of(".");

    if (lastdot == std::string::npos) return filename;
    return filename.substr(0, lastdot);
}

// Build output filename
std::string build_output_filename(const std::string &inputFilename,
                                  const BatchParams &params)
{
    std::string base = get_filename_without_extension(inputFilename);
    std::string output = base + "_bs" + std::to_string(params.blurStd);

    if (params.useAdaptive) {
        output += "_adaptive";
    } else {
        output += "_th" + std::to_string(params.threshold1);
        output += "_th" + std::to_string(params.threshold2);
    }

    if (params.hystIters == 0) {
        output += "_nohyst";
    }

    output += ".png";
    return output;
}

// Print batch processing summary
void print_batch_summary(const std::vector<std::string> &files, const BatchParams &params)
{
    std::cout << "\n========== Batch Processing Summary ==========" << std::endl;
    std::cout << "Input directory:  " << params.inputDir << std::endl;
    std::cout << "Output directory: " << params.outputDir << std::endl;
    std::cout << "Number of files:  " << files.size() << std::endl;
    std::cout << "Blur stdev:       " << params.blurStd << std::endl;
    std::cout << "Adaptive mode:    " << (params.useAdaptive ? "Yes" : "No") << std::endl;

    if (!params.useAdaptive) {
        std::cout << "Threshold 1:      " << params.threshold1 << std::endl;
        std::cout << "Threshold 2:      " << params.threshold2 << std::endl;
    }

    std::cout << "Hysteresis iters: " << params.hystIters << std::endl;
    std::cout << "Sync mode:        " << (params.doSync ? "Yes" : "No") << std::endl;
    std::cout << "=============================================\n" << std::endl;
}
