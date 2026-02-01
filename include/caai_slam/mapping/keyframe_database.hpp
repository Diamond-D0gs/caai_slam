#pragma once

#include "caai_slam/frontend/keyframe.hpp"
#include "caai_slam/core/types.hpp"

#include <unordered_map>
#include <shared_mutex>
#include <vector>
#include <memory>

namespace caai_slam {
    /**
     * @brief Global storage for all keyframes in the system
     * 
     * Maintains the persistent history of the SLAM session
     * Responsibilities:
     * - Storing keyframes that have been marginalized out of the local map
     * - Providing O(1) retrieval by ID
     * - Supporting global trajectory operations (saving, visualization)
     */
    class keyframe_database {
    private:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        // Core storage: ID -> keyframe
        std::unordered_map<uint64_t, std::shared_ptr<keyframe>> keyframes;
        uint64_t last_id = 0;

        // Thread safety
        mutable std::shared_mutex mutex;

    public:
        /**
         * @brief Add a keyframe to the global database
         * 
         * @param kf Keyframe to be added to the global database
         */
        void add(const std::shared_ptr<keyframe>& kf);

        /**
         * @brief Remove a keyframe by ID
         * 
         * @param id ID of the keyframe to be removed
         */
        void remove(const uint64_t id);

        /**
         * @brief Retrieve a keyframe by ID
         * 
         * @param id ID of the keyframe to be retrieved
         * 
         * @return nullptr if not found
         */
        std::shared_ptr<keyframe> get(const uint64_t id) const;

        /**
         * @brief Check if a keyframe exists in the database
         * 
         * @param id ID of the keyframe to check exists in the database
         * 
         * @return True if keyframe exists in the database
         */
        bool contains(const uint64_t) const;

        /**
         * @brief Retrieve all keyframes (e.g., for saving trajectory)
         * 
         * @return A vector of all keyframes, sorted by ID
         */
        std::vector<std::shared_ptr<keyframe>> get_all_keyframes() const;

        /**
         * @brief Get the latest keyframe ID stored
         * 
         * @return ID of the latest keyframe stored
         */
        uint64_t get_last_id() const;

        /**
         * @brief Get total number of keyframes in the database
         * 
         * @return Total number of keyframes in the database
         */
        size_t size() const;

        /**
         * @brief Clear all entries in the database
         */
        void clear();
    };

} // namespace caai_slam