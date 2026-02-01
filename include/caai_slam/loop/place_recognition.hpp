#pragma once

#include "caai_slam/frontend/keyframe.hpp"
#include "caai_slam/core/config.hpp"
#include "caai_slam/core/types.hpp"

#include <fbow.h>

#include <unordered_map>
#include <shared_mutex>
#include <vector>
#include <memory>
#include <mutex>

namespace caai_slam {
    /**
     * @brief Manages the Bag-of-Words database for apperance based loop detection
     * 
     * Uses the FBoW library to convert keyframe descriptors into visual words
     * and maintains an inverted index for fast retrieval of candidate matches.
     */
    class place_recognition {
    private:
        config _config;
        
        fbow::Vocabulary vocab;

        // Inverted Index: word ID -> list of keyframes containing the word
        std::unordered_map<uint32_t, std::vector<std::shared_ptr<keyframe>>> inverted_index;

        // Thread safety
        mutable std::shared_mutex mutex;

    public:
        /**
         * @param cfg System configuration for similarity threshold and exclusion logic
         */
        explicit place_recognition(const config& cfg) : _config(cfg) {}

        /**
         * @brief Load the pre-trained FBoW vocabulary file
         * 
         * @param path Path to the binary vocabulary file (.fbow)
         * 
         * @return True if the vocabulary was loaded and is valid
         */
        bool load_vocabulary(const std::string& path);

        /**
         * @brief Index a keyframe in the database
         * 
         * @param kf Keyframe contained extracted descriptors
         */
        void add_keyframe(const std::shared_ptr<keyframe>& kf);

        /**
         * @brief Find past keyframes that are visually similar to the query
         * 
         * @param kf The query keyframe
         * @param max_results Maximum number of candidates to return
         * 
         * @return List of candidates sorted by similarity
         */
        std::vector<std::shared_ptr<keyframe>> query(const std::shared_ptr<keyframe>& kf, const uint32_t max_results = 5);

        /**
         * @brief Clear all index keyframes from the database
         */
        void clear();
    };

} // namespace caai_slam