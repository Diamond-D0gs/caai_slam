#include "caai_slam/mapping/keyframe_database.hpp"

#include <algorithm>

namespace caai_slam {
    void keyframe_database::add(const std::shared_ptr<keyframe>& kf) {
        if (!kf)
            return;

        std::unique_lock<std::shared_mutex> lock(mutex);

        // Prevent duplicate insertion
        if (keyframes.find(kf->id) == keyframes.end()) {
            keyframes[kf->id] = kf;
            if (kf->id > last_id)
                last_id = kf->id;
        }
    }

    void keyframe_database::remove(const uint64_t id) {
        std::unique_lock<std::shared_mutex> lock(mutex);
        keyframes.erase(id);
    }

    std::shared_ptr<keyframe> keyframe_database::get(const uint64_t id) const {
        std::shared_lock<std::shared_mutex> lock(mutex);

        auto it = keyframes.find(id);
        if (it == keyframes.end())
            return nullptr;

        return it->second;
    }

    bool keyframe_database::contains(const uint64_t id) const {
        std::shared_lock<std::shared_mutex> lock(mutex);
        return keyframes.find(id) != keyframes.end();
    }

    std::vector<std::shared_ptr<keyframe>> keyframe_database::get_all_keyframes() const {
        std::shared_lock<std::shared_mutex> lock(mutex);

        std::vector<std::shared_ptr<keyframe>> all_kfs;
        all_kfs.reserve(keyframes.size());

        for (const auto& [unused, kf] : keyframes)
            all_kfs.push_back(kf);

        // Return sorted by ID for consistent processing order (e.g. trajactory saving)
        std::sort(all_kfs.begin(), all_kfs.end(), [](const auto& a, const auto& b) { return a->id < b->id; });

        return all_kfs;
    }

    uint64_t keyframe_database::get_last_id() const {
        std::shared_lock<std::shared_mutex> lock(mutex);
        return last_id;
    }

    size_t keyframe_database::size() const {
        std::shared_lock<std::shared_mutex> lock(mutex);
        return keyframes.size();
    }

    void keyframe_database::clear() {
        std::unique_lock<std::shared_mutex> lock(mutex);
        keyframes.clear();
        last_id = 0;
    }

} // namespace caai_slam