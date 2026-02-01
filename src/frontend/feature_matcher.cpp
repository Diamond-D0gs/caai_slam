#include "caai_slam/frontend/feature_matcher.hpp"

namespace caai_slam {
    feature_matcher::feature_matcher(const config& cfg) : _config(cfg) {
        // AKAZE descriptors are binary (M-LDB), so NORM_HAMMING is required.
        // crossCheck is set to false because we performa  manual knnMatch + ratio test.
        matcher = cv::BFMatcher::create(cv::NORM_HAMMING, false);
    }

    std::vector<cv::DMatch> feature_matcher::match(const cv::Mat& descriptors_query, const cv::Mat& descriptors_train) {
        if (descriptors_query.empty() || descriptors_train.empty() || descriptors_train.rows < 2) // We require at least 2 neighbors for the ratio test
            return;

        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(descriptors_query, descriptors_train, knn_matches, 2);

        // Apply Lowe's ratio test
        std::vector<cv::DMatch> out_matches;
        out_matches.reserve(knn_matches.size());
        for (const auto& m : knn_matches)
            if (m.size() >= 2 && m[0].distance < (_config.frontend.match_ratio_thresh * m[1].distance))
                out_matches.push_back(m[0]);

        return out_matches;
    }

} // namespace caai_slam