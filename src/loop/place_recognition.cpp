#include "caai_slam/loop/place_recognition.hpp"

#include <algorithm>
#include <iostream>

namespace caai_slam {
    bool place_recognition::load_vocabulary(const std::string& path) {
        std::unique_lock<std::shared_mutex> lock(mutex);

        try {
            vocab.readFromFile(path);
        }
        catch (const std::exception& e) {
            std::cerr << "PlaceRecognition: Failed to load vocabulary: " << e.what() << std::endl;
            return false;
        }

        return vocab.isValid();
    }

    void place_recognition::add_keyframe(const std::shared_ptr<keyframe>& kf) {
        if (!kf || !vocab.isValid())
            return;

        std::unique_lock<std::shared_mutex> lock(mutex);

        // Ensure BoW vectors are computed
        if (kf->bow_vec.empty())
            kf->compute_bow(vocab);

        // Add to inverted index for query acceleration
        for (const auto& [word_idx, unused] : kf->bow_vec)
            inverted_index[word_idx].push_back(kf);
    }

    std::vector<std::shared_ptr<keyframe>> place_recognition::query(const std::shared_ptr<keyframe>& kf, const uint32_t max_results) {
        if (!kf || kf->bow_vec.empty())
            return {};

        std::shared_lock<std::shared_mutex> lock(mutex);

        if (!vocab.isValid())
            return {};

        // 1. Collect candidates sharing words with the query
        std::unordered_map<keyframe*, uint32_t> candidate_counts;
        for (const auto& [word_idx, word_weight] : kf->bow_vec) {
            auto it = inverted_index.find(word_idx);
            if (it != inverted_index.end())
                for (const auto& candidate : it->second)
                    // Exclusion logic: skip immediate temporal neighbors
                    if (kf->id <= candidate->id || (kf->id - candidate->id) >= static_cast<uint64_t>(_config.loop.exclude_recent_n))
                        candidate_counts[candidate.get()]++;
        }

        // 2. Score candidates using visual similarity
        std::vector<std::pair<double, std::shared_ptr<keyframe>>> scored_candidates;
        for (const auto& [candidate_ptr, candidate_count] : candidate_counts)
            // Only score candidates sharing a minimum number of words
            if (candidate_count >= 5U) {
                const double score = fbow::fBow::score(kf->bow_vec, candidate_ptr->bow_vec);
                if (score >= static_cast<double>(_config.loop.similiarity_threshold))
                    scored_candidates.emplace_back(score, candidate_ptr->shared_from_this());
            }

        // 3. Sort by similarity score descending
        std::sort(scored_candidates.begin(), scored_candidates.end(), [](const auto& a, const auto& b) { return a.first > b.first; });

        // 4. Extract top results
        const uint32_t count = std::min(max_results, static_cast<uint32_t>(scored_candidates.size()));

        std::vector<std::shared_ptr<keyframe>> results;
        results.reserve(count);

        for (auto i = 0; i < count; ++i)
            results.push_back(scored_candidates[i].second);

        return results;
    }

    void place_recognition::clear() {
        std::unique_lock<std::shared_mutex> lock(mutex);
        inverted_index.clear();
    }

} // namespace caai_slam