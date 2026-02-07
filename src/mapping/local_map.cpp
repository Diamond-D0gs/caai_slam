#include "caai_slam/mapping/local_map.hpp"

#include <algorithm>
#include <iostream>

namespace caai_slam {
    void local_map::add_keyframe(const std::shared_ptr<keyframe>& kf) {
        if (!kf)
            return;

        std::unique_lock<std::shared_mutex> lock(mutex);

        // 1. Add to active window.
        active_keyframes.push_back(kf);

        // 2. Register map points observed by this keyframe.
        // If the keyframe created new map points (triangulated), add them to the set.
        for (const auto& mp : kf->map_points)
            // Ensure bidirectional link is valid (handled by triangulation usually, but safe to check).
            // mp->add_observation(kf, idx); Assumed handled by frontend.
            if (mp)
                active_map_points.insert(mp);

        // 3. Update covisibility graph.
        // Links this new keyframe to the existing ones based on shared map points.
        covis_graph.update(kf);
    }

    void local_map::prune_old_keyframes(const double current_timestamp) {
        std::unique_lock<std::shared_mutex> lock(mutex);

        const double lag_threshold = current_timestamp - _config.backend.lag_time;

        // GTSAM's FixedLagSmoother marginalizes old states.
        // We must remove them from the active window to prevent memory overgrowth.
        while (!active_keyframes.empty()) {
            const auto& oldest_kf = active_keyframes.front();

            // Check time condition (allow small buffer).
            if (oldest_kf->_timestamp > lag_threshold)
                break;

            // Removal process

            // 1. Remove observations from map points.
            for (auto& mp : oldest_kf->map_points)
                if (mp) {
                    mp->remove_observation(oldest_kf);

                    // If map point has no more observations, we remove it from the active set.
                    // Strictly speaking, it might still exist in the graph optimization until it is culled there,
                    // but for the local map's tracking purposes, it is dead.
                    if (mp->get_observations().empty())
                        active_map_points.erase(mp);
                }

            // 2. Remove from covisibility graph
            covis_graph.remove_keyframe(oldest_kf);

            // 3. Archive to the keyframe database for loop closure.
            // TODO: Implement

            // 4. Remove from deque
            active_keyframes.pop_front();
        }
    }

    void local_map::add_map_point(const std::shared_ptr<map_point>& mp) {
        if (!mp)
            return;

        std::unique_lock<std::shared_mutex> lock(mutex);

        active_map_points.insert(mp);
    }

    void local_map::fuse_map_points(std::shared_ptr<map_point>& target, std::shared_ptr<map_point>& victim) {
        if (!target || !victim || target == victim)
            return;

        std::vector<std::pair<std::shared_ptr<keyframe>, size_t>> moved_observations;

        {
            std::unique_lock<std::shared_mutex> lock(mutex);

            const auto& victim_observations = victim->get_observations();
            
            moved_observations.reserve(victim_observations.size());

            // Move observations from victim to target.
            for (const auto& [kf, feat_id] : victim_observations)
                if (kf)
                    moved_observations.emplace_back(kf, feat_id);

            // Mark victim as bad so it gets cleaned up elsewhere if referenced.
            victim->is_bad = true;

            // Remove victim from the active set.
            active_map_points.erase(victim);
        }

        for (const auto& [kf, feat_id] : moved_observations) {
            // Update keyframe to point to 'target' instead of 'victim'.
            std::lock_guard<std::mutex> kf_lock(kf->mutex);
            if (feat_id < kf->map_points.size()) {
                kf->map_points[feat_id] = target;
                target->add_observation(kf, feat_id);
            }
        }
    }

    void local_map::cull_map_points() {
        std::unique_lock<std::shared_mutex> lock(mutex);

        auto it = active_map_points.begin();
        while (it != active_map_points.end()) {
            const std::shared_ptr<map_point>& mp = *it;

            // Criteria for culling:
            // 1. Check the explicit 'bad' flag.
            // 2. Check the observation count.
            // PATCH: Increased minimum observations to 3 to cull underconstrained points earlier
            if (mp->is_bad || mp->get_observation_count() < 3) {
                // Nullify references in keyframes to prevent dangling pointers.
                for (const auto& [kf, feat_id] : mp->get_observations())
                    if (kf) {
                        std::lock_guard<std::mutex> kf_lock(kf->mutex);
                        if (feat_id < kf->map_points.size())
                            kf->map_points[feat_id] = nullptr;
                    }

                mp->is_bad = true; // Ensure 'bad' flag is set.
                
                it = active_map_points.erase(it);
            }
            else
                ++it;
        }
    }

    std::vector<std::shared_ptr<map_point>> local_map::get_map_points_in_view(const se3& pose_world_cam) const {
        std::shared_lock<std::shared_mutex> lock(mutex);

        std::vector<std::shared_ptr<map_point>> visible_points;
        visible_points.reserve(active_map_points.size());

        // Invert pose: t_cam_world = t_world_cam^-1
        const se3 t_cam_world = pose_world_cam.inverse();

        for (const auto& mp : active_map_points)
            if (!mp->is_bad && is_point_in_frustrum(mp->position, t_cam_world))
                visible_points.push_back(mp);

        return visible_points;
    }

    bool local_map::is_point_in_frustrum(const vec3& pos_world, const se3& t_cam_world) const {
        // 1. Transform to camera frame
        const vec3 p_cam = t_cam_world * pos_world;

        // 2. Check depth (must be in front of the camera)
        const double min_dist = 0.1;
        const double max_dist = 40.0; // Could be made configurable

        if (p_cam.z() < min_dist || p_cam.z() > max_dist)
            return false;

        // 3. Project to image
        const vec2 uv = _config.camera.project(p_cam);

        // 4. Check image bounds (with margin)
        const double margin = 10.0;

        if (uv.x() < -margin || uv.x() > _config.camera.width + margin || uv.y() < -margin || uv.y() > _config.camera.height + margin)
            return false;

        return true;
    }

    std::vector<std::shared_ptr<keyframe>> local_map::get_all_keyframes() const {
        std::shared_lock<std::shared_mutex> lock(mutex);
        return {active_keyframes.begin(), active_keyframes.end()};
    }

    std::vector<std::shared_ptr<keyframe>> local_map::get_covisible_keyframes(const std::shared_ptr<keyframe>& kf, const int32_t min_shared_points) const {
        return covis_graph.get_connected_keyframes(kf, min_shared_points);
    }

    size_t local_map::num_keyframes() const {
        std::shared_lock<std::shared_mutex> lock(mutex);
        return active_keyframes.size();
    }

    size_t local_map::num_map_points() const {
        std::shared_lock<std::shared_mutex> lock(mutex);
        return active_map_points.size();
    }

} // namespace caai_slam