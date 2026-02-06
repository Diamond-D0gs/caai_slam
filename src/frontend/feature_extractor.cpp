#include "caai_slam/frontend/feature_extractor.hpp"

#include <algorithm>

namespace caai_slam {
    feature_extractor::feature_extractor(const config& cfg) : _config(cfg) {
        // Initialize AKAZE with parameters from config
        // Descriptor type: M-LDB (AKAZE default)
        akaze = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0 /* descriptor size */, 3 /* descriptor channels */, cfg.frontend.akaze_threshold, 4 /* n octaves */, 4 /* n octave layers */, cv::KAZE::DIFF_PM_G2 /* diffusivity */);
    }

    void feature_extractor::limit_features(std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, const uint32_t max_features) {
        if (keypoints.empty())
            return;

        // Bounds check
        const size_t keep_n = std::min(static_cast<size_t>(max_features), keypoints.size());
        if (keep_n == keypoints.size())
            return;

        // Sort by response (strength of the feature)
        std::vector<int32_t> indices(keypoints.size());
        for (size_t i = 0; i < indices.size(); ++i)
            indices[i] = static_cast<int32_t>(i);

        std::sort(indices.begin(), indices.end(), [&](int32_t a, int32_t b) { return keypoints[a].response > keypoints[b].response; });

        // Retain top features
        cv::Mat top_descriptors;
        std::vector<cv::KeyPoint> top_keypoints;
        top_keypoints.reserve(keep_n);

        // Descriptors is a row-major matrix: rows = keypoints, cols = descriptor size
        top_descriptors.create(keep_n, descriptors.cols, descriptors.type());

        for (auto i = 0; i < keep_n; ++i) {
            top_keypoints.push_back(keypoints[indices[i]]);
            descriptors.row(indices[i]).copyTo(top_descriptors.row(i));
        }

        keypoints = std::move(top_keypoints);
        descriptors = std::move(top_descriptors);
    }

    void feature_extractor::detect_and_compute(const cv::Mat& image, std::vector<cv::KeyPoint>& out_keypoints, cv::Mat& out_descriptors) {
        if (image.empty())
            return;

        // Perform detection and description
        akaze->detectAndCompute(image, cv::noArray(), out_keypoints, out_descriptors);

        // Limit the number of features if they exceed the configured maximum
        if (static_cast<int32_t>(out_keypoints.size()) > _config.frontend.max_features)
            limit_features(out_keypoints, out_descriptors, _config.frontend.max_features);
    }

} // namespace caai_slam