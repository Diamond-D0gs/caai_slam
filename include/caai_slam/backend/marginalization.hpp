#pragma once

#include "caai_slam/core/types.hpp"

#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>

#include <mutex>

namespace caai_slam {
    /**
     * @brief Utility class to compute and analyze marginal covariances.
     * 
     * In graph-based SLAM, marginalization usually refers to removing old states.
     * GTSAM's FixedLagSmoother handles the removal automatically.
     * 
     * This class handles the analysis of the marginals:
     * 1. Recovering the covariance (uncertainty) of specific poses/landmarks.
     * 2. Calculating information entropy (useful for active keyframing).
     */
    class marginalization {
    private:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        // Cache the marginals object because construction is expensive.
        std::unique_ptr<gtsam::Marginals> marginals_computer;

        bool is_computed = false;

        // Thread safety
        mutable std::mutex mutex;

    public:
        /**
         * @brief Compute marginal covariances for the given graph and values.
         * 
         * This is an expensive operation (inverts the information matrix).
         * 
         * @param graph Input non-linear factor graph.
         * @param values Input values.
         */
        void compute(const gtsam::NonlinearFactorGraph& graph, const gtsam::Values& values);

        /**
         * @brief Get the 6x6 covariance matrix for a pose (rotation + position).
         * 
         * @param kf_id ID of the keyframe to get the covariance of the pose from.
         * 
         * @return Covariance matrix of the keyframe's pose.
         */
        mat6 get_pose_covariance(const uint64_t kf_id) const;

        /**
         * @brief Get the 3x3 covariance matrix for a landmark (position).
         * 
         * @param mp_id ID of the map point to get the covariance of the landmark from.
         * 
         * @return Covariance matrix of the map point's landmark.
         */
        mat3 get_landmark_covariance(const uint64_t mp_id) const;

        /**
         * @brief Get the 3x3 covariance for velocity.
         * 
         * @param kf_id ID of the keyframe to get the covariance of its velocity from.
         * 
         * @return Covariance matrix of the keyframe's velocity.
         */
        mat3 get_velocity_covariance(const uint64_t kf_id) const;

        /**
         * @brief Calculate the Shannon Entropy of a pose (measure of uncertainty).
         * 
         * Higher entropy = more uncertain.
         * 
         * @param kf_id The keyframe whose pose's Shannon Entropy is being calculated.
         * 
         * @return The Shannon Entropy of the keyframe's pose.
         */
        double get_pose_entropy(const uint64_t kf_id) const;

        /**
         * @brief Clear cached data.
         */
        void clear();
    };

} // namespace caai_slam