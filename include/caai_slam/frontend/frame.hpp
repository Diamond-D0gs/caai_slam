#pragma once

#include "caai_slam/frontend/keyframe.hpp"
#include "caai_slam/core/types.hpp"

#include <opencv2/core.hpp>

#include <vector>
#include <memory>

namespace caai_slam {
    /**
     * @brief Represents a single captured image with extracted features and a pose estimate
     * 
     * Unlike a keyframe, a frame is a transient object used for tracking.
     * It holds the temporal state of the system before a keyframe decision is made.
     * 
     * TODO: Thread safety and proper reseting and serialization of 'next_id'.
     */
    struct frame {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        static std::atomic<uint64_t> next_id;

        uint64_t id;
        timestamp _timestamp;

        // Visual data
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;

        // Pose data
        se3 pose; // t_world_imu: transformation from IMU to world frame
        imu_bias bias; // Current IMU bias estimates (accel and gyro)
        vec3 velocity; // Velocity in the world frame

        // Feature-to-map associations
        // Indices correspond to the keypoints vector
        std::vector<std::shared_ptr<map_point>> map_points;

        /**
         * @param ts The acquisition timestamp of the image
         * @param kps Detected keypoints
         * @param descs Computed descriptors 
         */
        frame(timestamp ts, const std::vector<cv::KeyPoint>& kps, const cv::Mat& descs);

        /**
         * @brief Check if a specific feature index has an associated 3D map point
         * 
         * @param idx Index of the keypoint
         * 
         * @return True if an association exists and the point is not marked as bad
         */
        bool has_map_point(const size_t idx) const;

        /**
         * @brief Compute the camera center in the world coordinates
         * 
         * @param t_cam_imu Extrinsics: transformation from IMU to camera frame
         * 
         * @return The 3D position of the camera center in the world frame
         */
        vec3 get_camera_center(const se3& t_cam_imu) const;
    };

} // namespace caai_slam