#pragma once

#include "caai_slam/core/config.hpp"
#include "caai_slam/core/types.hpp"

#include <opencv2/features2d.hpp>

#include <memory>
#include <vector>

namespace caai_slam {
    /**
     * @brief Wrapper for OpenCV AKAZE feature extraction
     * 
     * Handles the detection of keypoints and computation of descriptors
     * from grayscale images using the AKAZE algorithm.
     */
    class feature_extractor {
    private:
        config _config;

        cv::Ptr<cv::AKAZE> akaze;

        /**
         * @brief Filter and limit the number keypoints based on response strength
         * 
         * @param keypoints Input/output vector of keypoints
         * @param descriptors Input/output matrix of descriptors
         * @param max_features Maximum number of features to keep
         */
        void limit_features(std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, const uint32_t max_features);

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        /**
         * @param cfg System configuration containing AKAZE thresholds and limits.
         */
        explicit feature_extractor(const config& cfg);

        /**
         * @brief Detect keypoints and compute descriptors for a given imageh
         * 
         * @param image Input grayscale image
         * @param out_keypoints Output vector of detected OpenCV keypoints
         * @param out_descriptors Output cv::Mat containing descriptors (one per row)
         */
        void detect_and_compute(const cv::Mat& image, std::vector<cv::KeyPoint>& out_keypoints, cv::Mat& out_descriptors);
    };

} // namespace caai_slam