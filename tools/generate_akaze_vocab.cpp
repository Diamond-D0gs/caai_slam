/**
 * @file generate_akaze_vocab.cpp
 * @brief Generate an FBoW vocabulary from AKAZE descriptors extracted from a dataset.
 *
 * Usage:
 *   generate_akaze_vocab <image_dir> <output_vocab.fbow> [options]
 *
 * Options:
 *   --k <int>           Branching factor (default: 10)
 *   --L <int>           Tree depth, -1 for unlimited (default: -1)
 *   --threads <int>     Number of threads (default: 4)
 *   --max-images <int>  Max images to process, 0 for all (default: 0)
 *   --max-iters <int>   Max k-means iterations (default: 11)
 *   --akaze-thresh <f>  AKAZE detector threshold (default: 0.001)
 *   --max-features <int> Max features per image (default: 500)
 *
 * The tool recursively scans <image_dir> for .png and .jpg files,
 * extracts AKAZE descriptors, and trains an FBoW vocabulary.
 *
 * Example (EuRoC dataset):
 *   generate_akaze_vocab /datasets/MH_01/mav0/cam0/data akaze_vocab.fbow --k 10 --max-images 500
 */

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

#include <vocabulary_creator.h>
#include <fbow.h>

#include <filesystem>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <chrono>

namespace fs = std::filesystem;

// ============================================================================
// Helpers
// ============================================================================

struct cli_options {
    std::string image_dir;
    std::string output_path = "akaze_vocab.fbow";
    int k = 10;
    int L = -1;            // unlimited depth
    int threads = 4;
    int max_images = 0;    // 0 = all
    int max_iters = 11;
    float akaze_threshold = 0.001f;
    int max_features = 500;
};

static void print_usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " <image_dir> <output_vocab.fbow> [options]\n"
        << "\nOptions:\n"
        << "  --k <int>            Branching factor (default: 10)\n"
        << "  --L <int>            Tree depth, -1 unlimited (default: -1)\n"
        << "  --threads <int>      Thread count (default: 4)\n"
        << "  --max-images <int>   Max images to load, 0=all (default: 0)\n"
        << "  --max-iters <int>    K-means max iterations (default: 11)\n"
        << "  --akaze-thresh <f>   AKAZE threshold (default: 0.001)\n"
        << "  --max-features <int> Max features per image (default: 500)\n";
}

static bool parse_args(int argc, char** argv, cli_options& opts) {
    if (argc < 3) { print_usage(argv[0]); return false; }
    opts.image_dir = argv[1];
    opts.output_path = argv[2];
    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--k" && i + 1 < argc)              opts.k = std::stoi(argv[++i]);
        else if (arg == "--L" && i + 1 < argc)          opts.L = std::stoi(argv[++i]);
        else if (arg == "--threads" && i + 1 < argc)    opts.threads = std::stoi(argv[++i]);
        else if (arg == "--max-images" && i + 1 < argc)  opts.max_images = std::stoi(argv[++i]);
        else if (arg == "--max-iters" && i + 1 < argc)   opts.max_iters = std::stoi(argv[++i]);
        else if (arg == "--akaze-thresh" && i + 1 < argc) opts.akaze_threshold = std::stof(argv[++i]);
        else if (arg == "--max-features" && i + 1 < argc) opts.max_features = std::stoi(argv[++i]);
        else { std::cerr << "Unknown option: " << arg << "\n"; print_usage(argv[0]); return false; }
    }
    return true;
}

static std::vector<std::string> collect_images(const std::string& dir, int max_images) {
    std::vector<std::string> paths;
    for (const auto& entry : fs::recursive_directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        auto ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" || ext == ".tif" || ext == ".tiff")
            paths.push_back(entry.path().string());
    }
    std::sort(paths.begin(), paths.end());
    if (max_images > 0 && paths.size() > static_cast<size_t>(max_images))
        paths.resize(max_images);
    return paths;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    cli_options opts;
    if (!parse_args(argc, argv, opts)) return 1;

    // ---- Collect images ----
    std::cout << "[vocab] Scanning: " << opts.image_dir << "\n";
    auto image_paths = collect_images(opts.image_dir, opts.max_images);
    if (image_paths.empty()) {
        std::cerr << "[vocab] ERROR: No images found in " << opts.image_dir << "\n";
        return 1;
    }
    std::cout << "[vocab] Found " << image_paths.size() << " images\n";

    // ---- Configure AKAZE ----
    auto akaze = cv::AKAZE::create();
    akaze->setThreshold(opts.akaze_threshold);

    // ---- Extract descriptors ----
    std::cout << "[vocab] Extracting AKAZE descriptors (threshold=" << opts.akaze_threshold
              << ", max_features=" << opts.max_features << ")...\n";

    std::vector<cv::Mat> all_descriptors;
    all_descriptors.reserve(image_paths.size());

    size_t total_features = 0;
    auto t_start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < image_paths.size(); ++i) {
        cv::Mat image = cv::imread(image_paths[i], cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            std::cerr << "[vocab] WARNING: Could not load " << image_paths[i] << ", skipping\n";
            continue;
        }

        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        akaze->detectAndCompute(image, cv::noArray(), keypoints, descriptors);

        if (descriptors.empty()) continue;

        // Limit features per image by response strength
        if (opts.max_features > 0 && descriptors.rows > opts.max_features) {
            // Sort keypoints by response (descending), keep top N
            std::vector<size_t> indices(keypoints.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::partial_sort(indices.begin(), indices.begin() + opts.max_features, indices.end(),
                [&keypoints](size_t a, size_t b) { return keypoints[a].response > keypoints[b].response; });
            
            cv::Mat filtered(opts.max_features, descriptors.cols, descriptors.type());
            for (int j = 0; j < opts.max_features; ++j)
                descriptors.row(static_cast<int>(indices[j])).copyTo(filtered.row(j));
            descriptors = filtered;
        }

        all_descriptors.push_back(descriptors);
        total_features += descriptors.rows;

        if ((i + 1) % 50 == 0 || i + 1 == image_paths.size())
            std::cout << "  [" << (i + 1) << "/" << image_paths.size() << "] "
                      << total_features << " features so far\n";
    }

    auto t_extract = std::chrono::high_resolution_clock::now();
    double extract_sec = std::chrono::duration<double>(t_extract - t_start).count();

    std::cout << "[vocab] Extraction complete: " << total_features << " features from "
              << all_descriptors.size() << " images in " << extract_sec << "s\n";

    if (all_descriptors.empty() || total_features < static_cast<size_t>(opts.k)) {
        std::cerr << "[vocab] ERROR: Insufficient features to build vocabulary\n";
        return 1;
    }

    // ---- Train vocabulary ----
    std::cout << "[vocab] Training vocabulary (k=" << opts.k << ", L=" << opts.L
              << ", threads=" << opts.threads << ", max_iters=" << opts.max_iters << ")...\n";

    fbow::VocabularyCreator creator;
    fbow::VocabularyCreator::Params params(
        static_cast<uint32_t>(opts.k),
        opts.L,
        static_cast<uint32_t>(opts.threads),
        opts.max_iters
    );
    params.verbose = true;

    fbow::Vocabulary vocab;
    auto t_train_start = std::chrono::high_resolution_clock::now();
    creator.create(vocab, all_descriptors, "akaze", params);
    auto t_train_end = std::chrono::high_resolution_clock::now();
    double train_sec = std::chrono::duration<double>(t_train_end - t_train_start).count();

    std::cout << "[vocab] Training complete in " << train_sec << "s\n";
    std::cout << "[vocab] Vocabulary info:\n"
              << "  Descriptor type: " << vocab.getDescName() << "\n"
              << "  Descriptor size: " << vocab.getDescSize() << " bytes\n"
              << "  Branching factor (k): " << vocab.getK() << "\n"
              << "  Total blocks: " << vocab.size() << "\n"
              << "  Hash: " << vocab.hash() << "\n";

    // ---- Save ----
    std::cout << "[vocab] Saving to: " << opts.output_path << "\n";
    vocab.saveToFile(opts.output_path);

    // ---- Verify by loading back ----
    fbow::Vocabulary verify_vocab;
    verify_vocab.readFromFile(opts.output_path);
    if (verify_vocab.isValid() && verify_vocab.hash() == vocab.hash()) {
        std::cout << "[vocab] Verification OK (hash match)\n";
    } else {
        std::cerr << "[vocab] WARNING: Verification failed!\n";
        return 1;
    }

    std::cout << "[vocab] Done. Total time: " << (extract_sec + train_sec) << "s\n";
    return 0;
}
