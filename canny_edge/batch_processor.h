#ifndef BATCH_PROCESSOR_H
#define BATCH_PROCESSOR_H

#include <string>
#include <vector>

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

// Function declarations
bool is_png_file(const std::string &filename);
std::vector<std::string> get_png_files(const std::string &directory);
bool ensure_directory_exists(const std::string &directory);
std::string get_filename_without_extension(const std::string &filename);
std::string build_output_filename(const std::string &inputFilename, const BatchParams &params);
void print_batch_summary(const std::vector<std::string> &files, const BatchParams &params);

#endif // BATCH_PROCESSOR_H
