#pragma once

#include "caai_slam/core/types.hpp"
#include "caai_slam/core/config.hpp"
#include "caai_slam/frontend/keyframe.hpp"

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/geometry/Cal3_S2.h>

#include <unordered_set>
#include <vector>
#include <mutex>

namespace caai_slam {
    /**
     * @brief Backend wrapper for GTSAM ISAM2 (Visual-Inertial Optimization).
     * 
     * Manages the factor graph containing:
     * - X(i): Poses
     * - V(i): Velocities
     * - B(i): IMU Biases
     * - l(j): Landmarks (Map Points)
     */
    class graph_optimizer {
    private:
        config _config;

        // GTSAM Core
        gtsam::ISAM2 isam;
        gtsam::NonlinearFactorGraph new_factors;
        gtsam::Values new_values;

        // Camera Calibration (GTSAM format)
        boost::shared_ptr<gtsam::Cal3_S2> camera_calibration; // Boost shared pointer used for legacy reasons w/ GTSAM.

        // Noise Models
        gtsam::noiseModel::Diagonal::shared_ptr pose_noise;
        gtsam::noiseModel::Diagonal::shared_ptr velocity_noise;
        gtsam::noiseModel::Diagonal::shared_ptr bias_noise;
        gtsam::noiseModel::Isotropic::shared_ptr visual_noise;

        // Bookkeeping
        bool pending_loop_closure = false;
        std::unordered_set<uint64_t> observed_landmarks; // IDs of landmarks already in the graph.
        state latest_state; // Cache of the latest optimization result.

        // Thread safety
        mutable std::mutex mutex;

        /**
         * @brief Compute full 15x15 state covariance from ISAM2.
         * 
         * Extracts marginal and joint covariances for pose, velocity, and bias, assembling them into a single
         * covariance matrix for use in state propagation and uncertainty visualization.
         * 
         * @param kf_id Keyframe ID to compute covariance for.
         * 
         * @return 15x15 covariance matrix.
         */
        Eigen::Matrix<double, 15, 15> compute_state_covariance(const uint64_t kf_id);

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        explicit graph_optimizer(const config& _config);

        /**
         * @brief Add the first keyframe with prior factors (pose, velocity, bias, etc.).
         * 
         * @param kf The first keyframe.
         * @param initial_state The initial state of the graph optimizer.
         */
        void add_first_keyframe(const std::shared_ptr<keyframe>& kf, const state& initial_state);

        /**
         * @brief Add a new keyframe with IMU and visual factors.
         * 
         * @param kf The new keyframe to add.
         * @param preintegrated_imu IMU measurements from last keyframe to this one.
         * @param previous_kf_id ID of the previous keyframe (for IMU linking).
         */
        void add_keyframe(const std::shared_ptr<keyframe>& kf, const gtsam::PreintegratedCombinedMeasurements& preintegrated_imu, uint64_t previous_kf_id);

        /**
         * @brief Add a loop closure constraint (BetweenFactor).
         * 
         * @param kf_id_from ID of the keyframe establishing the link.
         * @param kf_id_to ID of the keyframe being linked to.
         * @param rel_pose The pose between the two keyframes.
         */
        void add_loop_constraint(const uint64_t kf_id_from, const uint64_t kf_id_to, const se3& rel_pose);

        /**
         * @brief Perform ISAM2 optimization.
         * 
         * 1. Updates ISAM2 with new factors and values.
         * 2. Calculates best estimate.
         * 3. Updates keyframe poses and map point positions in memory.
         * 4. Returns the current estimated state (pose, velocity, bias, etc.).
         * 
         * @param curr_kf The current keyframe.
         * 
         * @return The current estimated state.
         */
        state optimize(std::shared_ptr<keyframe>& curr_kf, const std::vector<std::shared_ptr<keyframe>>& active_kfs);

        /**
         * @brief Get the latest optimized state.
         * 
         * @return The latest optimized state.
         */
        state get_last_state() const;
    };

} // namespace caai_slam