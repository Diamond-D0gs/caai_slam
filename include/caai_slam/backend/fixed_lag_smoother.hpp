#pragma once

#include "caai_slam/core/types.hpp"
#include "caai_slam/core/config.hpp"
#include "caai_slam/frontend/keyframe.hpp"

#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/nonlinear/Values.h>

#include <unordered_map>
#include <unordered_set>
#include <mutex>

namespace caai_slam {
    /**
     * @brief Backend optimization using GTSAM's IncrementalFixedLagSmoother.
     * 
     * Maintain's a sliding window of states. Old states are automatically
     * marginalized out via the Schur complement when they pass the lag horizon.
     */
    class fixed_lag_smoother {
    private:
        config _config;

        // GTSAM smoother
        // Note: IncrementalFixedLagSmoother uses ISAM2 internally for the active window.
        std::unique_ptr<gtsam::IncrementalFixedLagSmoother> smoother;

        // Buffers for the next update
        gtsam::FixedLagSmoother::KeyTimestampMap new_timestamps;
        gtsam::NonlinearFactorGraph new_factors;
        gtsam::Values new_values;
        
        // Camera calibration
        boost::shared_ptr<gtsam::Cal3_S2> calibration; // Legacy boost::shared_ptr required for GTSAM factors.

        // Noise models
        gtsam::noiseModel::Diagonal::shared_ptr velocity_noise;
        gtsam::noiseModel::Diagonal::shared_ptr bias_rw_noise;
        gtsam::noiseModel::Isotropic::shared_ptr visual_noise;
        gtsam::noiseModel::Diagonal::shared_ptr bias_noise;
        gtsam::noiseModel::Diagonal::shared_ptr pose_noise;
        
        // Random walk
        std::unordered_set<uint64_t> observed_landmarks;
        state latest_state;

        // Thread safety
        mutable std::mutex mutex;

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        explicit fixed_lag_smoother(const config& cfg);

        /**
         * @brief Initialize the graph with the first keyframe (priors).
         * 
         * @param kf Keyframe to initialize the graph with.
         * @param initial_state The initial state of the fixed lag smoother.
         */
        void initialize(const std::shared_ptr<keyframe>& kf, const state& initial_state);

        /**
         * @brief Add a new keyframe with IMU preintegration and visual observations.
         * 
         * @param kf Current keyframe
         * @param imu_meas Preintegrated IMU from previous keyframe to current keyframe.
         * @param prev_kf_id ID of the previous keyframe
         */
        void add_keyframe(const std::shared_ptr<keyframe>& kf, const gtsam::PreintegratedCombinedMeasurements& imu_meas, const uint64_t prev_kf_id);

        /**
         * @brief Trigger for the optimization step.
         * 
         * @return The set of keys that were marginalized out in this step (used to prune local map).
         */
        std::vector<uint64_t> optimize();

        /**
         * @brief Get the latest optimized state estimate.
         * 
         * @return The latest state of the fixed lag smoother.
         */
        state get_latest_state() const;

        /**
         * @brief Update the internal data of a specific keyframe with optimized values.
         * 
         * @param kf Keyframe to update with optimized values.
         */
        void update_keyframe_state(std::shared_ptr<keyframe>& kf);
    };

} // namespace caai_slam