#pragma once

#include "caai_slam/core/types.hpp"
#include "caai_slam/frontend/keyframe.hpp"

#include <unordered_map>
#include <shared_mutex>
#include <deque>
#include <set>

namespace caai_slam {
    /**
     * @brief Manages the weighted topological graph of keyframes.
     * 
     * Nodes: Keyframes
     * Edges: Number of shared map points (weights).
     * 
     * Used for:
     * 1. Defining local bundle adjustment window.
     * 2. Retrieving spatial neighbors for tracking.
     */
    class covisibility_graph {
    private:
        // Adjacency list: [KF_ID] -> map<neighbor_id, weight>
        std::unordered_map<uint64_t, std::unordered_map<uint64_t, int32_t>> adjacency_map;

        // Quick lookup for keyframe pointers by ID.
        std::unordered_map<uint64_t, std::shared_ptr<keyframe>> nodes;

        // Thread safety
        mutable std::shared_mutex mutex;

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        /**
         * @brief Update connections for a specific keyframe.
         * 
         * Called when a keyframe is created or its map points change (fusion).
         * 1. Identifies neighbors via shared map points.
         * 2. Filters neighbors with shared count < min_threshold (default 15).
         * 3. Updates the central adjacency graph.
         * 4. Updates the 'connected_keyframes' vector in the keyframe (sorted by weight).
         * 5. Updates the neighbors' connections back to this keyframe.
         */
        void update(std::shared_ptr<keyframe> kf, int32_t min_weight = 15);

        /**
         * @brief Remove a keyframe from the graph.
         * 
         * Removes the node and all edges connected to it.
         * Updates neighbors to remove their links to this keyframe.
         */
        void remove_keyframe(std::shared_ptr<keyframe> kf);

        /**
         * @brief Get neighbors of a keyframe with edge weight >= min_weight.
         */
        std::vector<std::shared_ptr<keyframe>> get_connected_keyframes(const std::shared_ptr<keyframe>& kf, int32_t min_weight = 15) const;

        /**
         * @brief Get the top N neighbors sorted by weight.
         */
        std::vector<std::shared_ptr<keyframe>> get_best_covisibility_keyframes(const std::shared_ptr<keyframe>& kf, size_t n) const;

        /**
         * @brief Get a connected component (BFS) starting from the keyframe.
         * Used to define the local BA window.
         */
        std::set<std::shared_ptr<keyframe>> get_connected_component(const std::shared_ptr<keyframe>& kf, int32_t depth = 2, int32_t min_weight = 15) const;

        /**
         * @brief Get the weight (shared points) between two keyframes.
         */
        int32_t get_weight(const std::shared_ptr<keyframe>& kf1, const std::shared_ptr<keyframe>& kf2) const;

        void clear();
    };
}