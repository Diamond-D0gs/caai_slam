#pragma once

#include "caai_slam/core/types.hpp"

#include <iostream>
#include <string>

namespace caai_slam {
    /**
     * @brief Global System Configuration
     * 
     * Loaded from a YAML file (e.g., EuRoC.yaml).
     * Holds parameters for all subsystems (frontend, backend, vio, loop closure, etc.).
     */
    struct config {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        // =========================================================================
        // Sensor Parameters
        // =========================================================================

        camera_intrinsics camera;
        extrinsics _extrinsics; // t_cam_imu, time_offset

        struct imu_params {
            // Continuous-time noise densities
            double accel_noise_density; // m/s^2 / sqrt(Hz)
            double gyro_noise_density; // rad/s / sqrt(Hz)

            // Random walk parameters
            double accel_random_walk; // m/s^3 / sqrt(Hz)
            double gyro_random_walk; // rad/s^2 / sqrt(Hz)

            double gravity_magnitude = 9.81;
            uint32_t frequency = 200; // Hz
        } imu;

        // =========================================================================
        // Algorithm Parameters
        // =========================================================================

        struct frontend_params {
            int32_t max_features = 1000;
            float akaze_threshold = 0.001f;
            float match_ratio_thresh = 0.8f; // Lowe's ratio test
            int32_t min_matches_tracking = 15; // Min inliers to survive tracking
            int32_t min_matches_init = 30; // Min matches to attempt init
            double parallax_min = 1.0; // Degrees
        } frontend;

        struct backend_params {
            //ISAM2 settings
            double relinearize_threshold = 0.1;
            int32_t relinearize_skip = 1;

            // Fixed-lag smoother settings
            double lag_time = 5.0; // Seconds of history to optimize.

            // Optimization noise (std devs)
            double loop_closure_noise_pos = 0.05; // meters
            double loop_closure_noise_rot = 0.05; // radians
        } backend;

        struct loop_params {
            bool enable = true;
            float similarity_threshold = 0.05f; // FBoW score threshold
            int32_t min_matches_geom = 12; // Min inliers for TEASER++
            int32_t exclude_recent_n = 20; // Don't loop close with immediate neighbors
        } loop;

        // =========================================================================
        // Methods
        // =========================================================================

        /**
         * @brief Load configuration from a YAML file.
         * Uses OpenCV FileStorage.
         * @return true if successful
         */
        bool loadFromYAML(const std::string& filename);

        // Helper to print loaded config for verification.
        void print() const;
    };

} // namespace caai_slam