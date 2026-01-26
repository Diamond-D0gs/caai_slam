#include "caai_slam/frontend/keyframe.hpp"

#include <iostream>

namespace caai_slam {
    // Static ID init.
    std::atomic<uint64_t> map_point::next_id{0};
    std::atomic<uint64_t> keyframe::next_id{0};

    // =============================================================================
    // Map Point Implementation
    // =============================================================================

    map_point::map_point(const vec3& pos, const cv::Mat& desc) 
        : id(next_id++), position(pos) { desc.copyTo(descriptor); }

    void map_point::add_observation(std::shared_ptr<keyframe> kf, size_t feature_idx) {
        std::lock_guard<std::mutex> lock(mutex);

        // Check if already observed by this keyframe.
        for (auto& observation : observations)
            if (observation.first.lock() == kf) {
                observation.second = feature_idx;
                return;
            }

        observations.emplace_back(kf, feature_idx);
    }

    void map_point::remove_observation(std::shared_ptr<keyframe> kf) {
        std::lock_guard<std::mutex> lock(mutex);

        // Erase-remove idiom.
        auto it = std::remove_if(observations.begin(), observations.end(),
            [&](const auto& observation) {
                return observation.first.lock() == kf;
            });

        if (it != observations.end())
            observations.erase(it, observations.end());

        if (observations.empty())
            is_bad = true; // Mark for culling if not seen.
    }

    std::map<std::shared_ptr<keyframe>, size_t> map_point::get_observations() const {
        std::lock_guard<std::mutex> lock(mutex);

        std::map<std::shared_ptr<keyframe>, size_t> valid_observations;

        for (const auto& observation : observations)
            if (auto kf = observation.first.lock())
                valid_observations[kf] = observation.second;

        return valid_observations;
    }

    // =============================================================================
    // Keyframe Implementation
    // =============================================================================

    void keyframe::compute_bow(fbow::Vocabulary& voc) {
        // FBoW transformation
        // Note: FBoW uses cv::Mat descriptors directly.
        if (!descriptors.empty())
            voc.transform(descriptors, 4, bow_vec, feat_vec);
    }

    void keyframe::set_pose(const gtsam::Pose3& p) {
        std::lock_guard<std::mutex> lock(mutex);
        pose = p;
    }

    gtsam::Pose3 keyframe::get_pose() const {
        std::lock_guard<std::mutex> lock(mutex);
        return pose;
    }

    std::vector<size_t> keyframe::get_feats_in_area(const float x, const float y, const float r) const {
        std::vector<size_t> indicies;
        indicies.reserve(keypoints.size());

        const float r_sq = r * r;

        for (size_t i = 0; i < keypoints.size(); ++i) {
            const auto& kp = keypoints[i];
            
            const float dx = kp.pt.x - x;
            const float dy = kp.pt.y - y;

            if (dx * dx + dy * dy < r_sq)
                indicies.push_back(i);
        }

        return indicies;
    }

} // namespace caai_slam