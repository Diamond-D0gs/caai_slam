#pragma once

#include <caai_slam/frontend/keyframe.hpp>

#include <teaser/registration.h>

#include <memory>

namespace caai_slam {
    /**
     * @brief Helper to convert OpenCV/Eigen types for TEASER++
     * 
     * @param kf1 Query keyframe
     * @param kf2 Candidate keyframe
     * @param src_cloud Output 3xN matrix of 3D points from kf1 (in kf1's IMU frame)
     * @param target_cloud Output 3xN matrix of 3D points from kf2 (in kf2's IMY frame)
     */
    void get_matched_points(const float match_ratio_thresh, const std::shared_ptr<keyframe>& query, const std::shared_ptr<keyframe>& candidate, Eigen::Matrix<double, 3, Eigen::Dynamic>& src_cloud, Eigen::Matrix<double, 3, Eigen::Dynamic>& target_cloud);

    inline uint32_t count_inliers(const double loop_closure_noise_pos, const teaser::RegistrationSolution& solution, const Eigen::Matrix<double, 3, Eigen::Dynamic>& src_cloud, const Eigen::Matrix<double, 3, Eigen::Dynamic>& target_cloud) {
        const mat3 r = solution.rotation;
        const vec3 t = solution.translation;

        const double max_sq_err = std::pow(loop_closure_noise_pos * 3.0, 2);
        
        uint32_t valid_inliers = 0;
        for (auto i = 0; i < src_cloud.cols(); ++i) 
            if (((r * src_cloud.col(i) + t) - target_cloud.col(i)).squaredNorm() < max_sq_err)
                ++valid_inliers;

        return valid_inliers;
    }

} // namespace caai_slam