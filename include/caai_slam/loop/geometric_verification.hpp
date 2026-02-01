#pragma once

#include "caai_slam/frontend/keyframe.hpp"
#include "caai_slam/core/config.hpp"
#include "caai_slam/core/types.hpp"

#include <teaser/registration.h>
#include <opencv2/core.hpp>

#include <memory>
#include <vector>

namespace caai_slam {
    /**
     * @brief Performs robust geometric validation between two keyframes
     * 
     * Uses TEASER++ for 3D-3D registration and OpenCV for descriptor based
     * correspondence verification to find the relative transformation.
     */
    class geometric_verification {
    private:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        config _config;

        std::unique_ptr<teaser::RobustRegistrationSolver> solver;

    public:
        /**
         * @brief Result of the geometric verification process
         */
        struct result {
            uint32_t inlier_count = 0; // Number of points satisfying the model
            bool success = false; // Whether a valid transformation was found
            se3 t_query_match; // The relative transformation (t_query_match)
        };

        /**
         * @param cfg System configuration for thresholds and noise bounds
         */
        explicit geometric_verification(const config& cfg);

        /**
         * @brief Verify the geometric consistency between two keyframes using 3D point clouds
         * 
         * @param query The query keyframe (source)
         * @param match The candidate match keyframe (target)
         * 
         * @return result Structure containing success flag, relative pose, and inlier count
         */
        result verify_3d_3d(const std::shared_ptr<keyframe>& query, const std::shared_ptr<keyframe>& match);
    };

} // namespace caai_slam