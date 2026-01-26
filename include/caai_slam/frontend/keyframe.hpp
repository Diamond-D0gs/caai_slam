#pragma once

#include "caai_slam/core/types.hpp"

#include <gtsam/inference/Symbol.h>
#include <gtsam/geometry/Pose3.h>
#include <fbow.h>

#include <mutex>
#include <map>

namespace caai_slam {
    struct keyframe; // Forward declaration

    /**
     * @brief 3D Landmark in the environment.
     */
    struct map_point {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        static std::atomic<uint64_t> next_id;

        // Identity
        uint64 id;
        // Position (Optimizable)
        vec3 position; // World frame
        // Appearance
        cv::Mat descriptor; // Representative descriptor (usually average of observations).
        // Topology: Keyframes observing this point.
        std::vector<std::pair<std::weak_ptr<keyframe>, size_t>> observations;
        // Optimization Flags
        bool is_bad = false; // To be culled
        uint64_t last_observed_frame_id = 0;
        // Thread safety
        mutable std::mutex mutex;

        map_point(const vec3& pos, const cv::Mat& desc);

        void add_observation(std::shared_ptr<keyframe> kf, size_t feature_idx);
        void remove_observation(std::shared_ptr<keyframe> kf);
        std::map<std::shared_ptr<keyframe>, size_t> get_observations() const;
        gtsam::Symbol symbol() const { return gtsam::Symbol('l', id); }
    };

    /**
     * @brief Keyframe containing pose, feature, and map_point associations.
     */
    struct keyframe : std::enable_shared_from_this<keyframe> {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        static std::atomic<uint64_t> next_id;

        // Identity
        uint64_t id;
        timestamp _timestamp;
        // Pose data (optimized by GTSAM).
        gtsam::Pose3 pose;
        // Visual Data
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors; // AKAZE descriptors (rows = keypoints)
        // Map associations (same size as keypoints).
        std::vector<std::shared_ptr<map_point>> map_points; // nullptr if feature is not triangulated.
        // Loop closure / place recognition.
        fbow::fBow bow_vec; 
        fbow::fBow2 feat_vec; // Direct index
        // Covisibility graph connections.
        std::vector<std::shared_ptr<keyframe>> connected_keyframes;
        std::vector<int32_t> connected_weights;
        // Thread safety
        mutable std::mutex mutex;

        keyframe::keyframe(double ts, const gtsam::Pose3& initial_pose)
            : id(next_id++), _timestamp(ts), pose(initial_pose) {}

        void compute_bow(fbow::Vocabulary& voc);
        void set_pose(const gtsam::Pose3& p);
        gtsam::Pose3 get_pose() const;
        std::vector<size_t> get_feats_in_area(const float x, const float y, const float r) const;
        gtsam::Symbol symbol() const { return gtsam::Symbol('x', id); }
    };

} // namespace caai_slam