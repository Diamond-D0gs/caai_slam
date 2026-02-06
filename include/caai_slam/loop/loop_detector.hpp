#pragma once

#include "caai_slam/frontend/keyframe.hpp"
#include "caai_slam/core/config.hpp"
#include "caai_slam/core/types.hpp"

#include <teaser/registration.h>
#include <fbow.h>

#include <unordered_map>
#include <memory>
#include <vector>
#include <map>

namespace caai_slam {
    /**
     * @brief Loop Closure Subsystem
     * 
     * Responsibilities:
     * 1. Place recognition (FBoW database)
     * 2. Geometric verification (TEASER 3D-3D registration)
     */
    class loop_detector {
    private:
        config _config;

        // FBoW vocabulary and database
        std::unique_ptr<fbow::Vocabulary> vocab;

        // Inverted index: word ID -> List of keyframes containing that word
        // Maps FBoW word index to a list of keyframes
        std::vector<std::vector<std::shared_ptr<keyframe>>> inverted_index;

        // All keys frame in the database (for ownership/lookup)
        std::vector<std::shared_ptr<keyframe>> database_keyframes;

        // TEASER++ solver
        std::unique_ptr<teaser::RobustRegistrationSolver> teaser_solver;

        // Thread safety
        mutable std::mutex mutex;

        // Internal methods

        /**
         * @brief Query FBow database for candidates
         * 
         * @param kf The current keyframe to query for
         * @param active_exclusion_list Active keyframes to exclude from matching (prevents self/neighbor matching)
         * 
         * @return A vector of candidate keyframes sorted by score (highest first)
         */
        std::vector<std::shared_ptr<keyframe>> query_database(const std::shared_ptr<keyframe>& kf, const std::vector<std::shared_ptr<keyframe>>& active_exclusion_list);

        /**
         * @brief Perform robust 3D-3D registration using TEASER++
         * 
         * @param query Keyframe to be checked against
         * @param candidate Keyframe that the query is checked against
         * @param out_t_cand_query Output relative pose (t_candidate query)
         * @param out_inliers Output number of inliers found by the solver
         * 
         * @return True if the registration is a successful match (valid solution & enough inliers)
         */
        bool verify_geometry(const std::shared_ptr<keyframe>& query, const std::shared_ptr<keyframe>& candidate, se3& out_t_cand_query, uint32_t& out_inliers);

    public:
        struct loop_result {
            std::shared_ptr<keyframe> query_kf, match_kf;
            bool is_detected = false;
            se3 t_match_query; // Transform that aligns query to match
            int inliers = 0;
        };

        explicit loop_detector(const config& cfg);

        /**
         * @brief Load the FBoW vocabulary from disk
         * 
         * Required before running detection
         * 
         * @param path Path to the FBoW vocabulary
         * 
         * @return True if the vocabulary was successfully loaded
         */
        bool load_vocabulary(const std::string& path);

        /**
         * @brief Add a keyframe to the loop database
         * 
         * Updates the inverted index
         * 
         * @param kf Keyframe to be added to the loop database
         */
        void add_keyframe(const std::shared_ptr<keyframe>& kf);

        /**
         * @brief Attemp to detect and verify a loop closure for the current keyframe
         * 
         * @param kf The current keyframe
         * @param active_kfs List of currently active keyframes (to exclude from matching)
         * 
         * @return loop_result containing match info and relative pose if successful
         */
        loop_result detect_loop(const std::shared_ptr<keyframe>& kf, const std::vector<std::shared_ptr<keyframe>>& active_kfs);

        /**
         * @brief Reset the database
         */
        void reset();
    };

} // namespace caai_slam