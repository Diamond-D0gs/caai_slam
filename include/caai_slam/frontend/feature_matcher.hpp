#pragma once

#include "caai_slam/core/config.hpp"
#include "caai_slam/core/types.hpp"

#include <opencv2/features2d.hpp>

#include <vector>

namespace caai_slam {
    /**
     * @brief Wrapper for OpenCV descriptor matching logic
     * 
     * Uses brute-force matching with hamming distance and impelements Lowe's ratio test for outlier rejection.
     */
    class feature_matcher {
    private:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        config _config;

        cv::Ptr<cv::BFMatcher> matcher;

    public:
        /**
         * @param cfg System configuration containing the matching ratio threshold
         */
        explicit feature_matcher(const config& cfg);

        /**
         * @brief Match descriptors between two sets using k-nearest neighbors and a ratio test
         * 
         * @param descriptors_query Matrix of descriptors from the query image
         * @param descriptors_train Matrix of descriptors from the train/reference image
         * 
         * @return Vector ontaining matches that passes the ratio test
         */
        std::vector<cv::DMatch> match(const cv::Mat& descriptors_query, const cv::Mat& descriptors_train);
    };

} // namespace caai_slam