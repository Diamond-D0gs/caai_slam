#include "caai_slam/mapping/covisibility_graph.hpp"

#include <algorithm>
#include <queue>

namespace caai_slam {
    // Helper to remove a specific keyframe from a neighbor's internal cache.
    void remove_kf_from_neighbor_cache(const std::shared_ptr<keyframe>& neighbor, const uint64_t kf_id) {
        std::lock_guard<std::mutex> lock(neighbor->mutex);

        auto& kfs = neighbor->connected_keyframes;
        auto& ws = neighbor->connected_weights;

        for (size_t i = 0; i < kfs.size(); ++i)
            if (kfs[i]->id == kf_id) {
                kfs.erase(kfs.begin() + i);
                ws.erase(ws.begin() + i);
                break;
            }
    }

    void covisibility_graph::update(std::shared_ptr<keyframe> kf, int32_t min_weight) {
        if (!kf)
            return;

        std::vector<std::pair<std::shared_ptr<keyframe>, int32_t>> valid_neighbors;
        // List of neighbors that need to be updated because they were dropped (count == 0).
        std::vector<std::shared_ptr<keyframe>> neighbors_to_disconnect;

        {
            std::unique_lock<std::shared_mutex> lock(mutex);

            nodes[kf->id] = kf;

            // Make a copy of previous edges to detect dropped connections (count == 0).
            auto previous_edges = adjacency_map[kf->id];

            // 1. Calculate weights (shared map points) with all other keyframes.
            std::map<std::shared_ptr<keyframe>, uint32_t> shared_counts;

            // Iterate through all map points observed by the keyframe.
            for (const auto& mp : kf->map_points)
                if (mp && !mp->is_bad)
                    // Get iterate observations of the current map point & update neighbor counts.
                    for (const auto& observation : mp->get_observations())
                        // Do not include self in the count.
                        if (observation.first->id != kf->id)
                            shared_counts[observation.first]++;

            // 2. Filter by threshold and prepare update data.
            // Only keep edges with weight >= min_weight.
            valid_neighbors.reserve(shared_counts.size());

            // Prepare to update the central adjacency map.
            auto& curr_edges = adjacency_map[kf->id];
            curr_edges.clear(); // Rebuild the edges for ->his node.

            for (const auto& shared_count : shared_counts) {
                const std::shared_ptr<keyframe>& neighbor = shared_count.first;
                const int32_t weight = shared_count.second;

                if (weight >= min_weight) {
                    valid_neighbors.emplace_back(neighbor, weight);

                    // Update central graph (Direction 1: Current -> Neighbor).
                    curr_edges[neighbor->id] = weight;

                    // Update central graph (Direction 2: Neighbor -> Current).
                    adjacency_map[neighbor->id][kf->id] = weight;
                }
                else {
                    // Explicitly remove weak edges from neighbor's record if they exist.
                    auto it = adjacency_map.find(neighbor->id);
                    if (it != adjacency_map.end())
                        it->second.erase(kf->id);

                    neighbors_to_disconnect.push_back(neighbor);
                }
            }

            // Handle dropped connections (count == 0)
            for (const auto& [prev_neighbor_id, prev_weight] : previous_edges) {
                // Only process if this neighbor is not in the current edges.
                if (curr_edges.count(prev_neighbor_id)) {
                    if (adjacency_map.count(prev_neighbor_id))
                        adjacency_map[prev_neighbor_id].erase(kf->id);
                    if (nodes.count(prev_neighbor_id))
                        neighbors_to_disconnect.push_back(nodes[prev_neighbor_id]);
                }
            }
        }
        

        // 3. Update the keyframe's internal cache (sorted descending by weight).
        std::sort(valid_neighbors.begin(), valid_neighbors.end(), [](const auto& a, const auto& b) { return a.second > b.second; }); // Descending

        {
            // Lock keyframe before modifying its members.
            std::lock_guard<std::mutex> kf_lock(kf->mutex);

            kf->connected_keyframes.clear();
            kf->connected_weights.clear();

            for (const auto& valid_neighbor : valid_neighbors) {
                kf->connected_keyframes.push_back(valid_neighbor.first);
                kf->connected_weights.push_back(valid_neighbor.second);
            }
        }

        // 4. Update the neighbors' internal caches.
        // Since the edge weights have been updated, the neighbors must re-sort/update their lists.
        // Optimization: Only update neighbors whose connection status actually changed. Update all valid neighbors.
        for (const auto& valid_neighbor : valid_neighbors) {
            const std::shared_ptr<keyframe>& neighbor = valid_neighbor.first;

            // This implementation will append and re-sort the neighbor
            std::lock_guard<std::mutex> n_lock(neighbor->mutex);

            auto& n_kfs = neighbor->connected_keyframes;
            auto& n_ws = neighbor->connected_weights;

            bool found = false;
            for (size_t i = 0; i < n_kfs.size(); ++i)
                if (n_kfs[i]->id == kf->id) {
                    n_ws[i] = valid_neighbor.second; // Update weight
                    found = true;
                    break;
                }

            if (!found) {
                n_kfs.push_back(kf);
                n_ws.push_back(valid_neighbor.second);
            }

            // Re-sort the neighbor's lists.
            std::vector<std::pair<int32_t, std::shared_ptr<keyframe>>> temp_sort;
            for (size_t i = 0; i < n_kfs.size(); ++i)
                temp_sort.emplace_back(n_ws[i], n_kfs[i]);

            std::sort(temp_sort.begin(), temp_sort.end(), [](const auto& a, const auto& b) {return a.first > b.first; });

            n_kfs.clear();
            n_ws.clear();

            for (const auto& p : temp_sort) {
                n_kfs.push_back(p.second);
                n_ws.push_back(p.first);
            }
        }

        // 5. Update disconnected neighbor's caches (remove).
        for (auto& neighbor : neighbors_to_disconnect)
            remove_kf_from_neighbor_cache(neighbor, kf->id);
    }

    void covisibility_graph::remove_keyframe(std::shared_ptr<keyframe> kf) {
        if (!kf)
            return;
        
        std::unique_lock<std::shared_mutex> lock(mutex);

        const uint64_t id = kf->id;

        // 1. Remove from all neighbors' adjacency lists.
        auto it = adjacency_map.find(id);
        if (it != adjacency_map.end())
            for (const auto& neighbor : it->second) {
                const uint64_t neighbor_id = neighbor.first;

                // Remove ID from neighbor's map.
                adjacency_map[neighbor_id].erase(id);

                // Remove keyframe from neighbor's internal stored vectors.
                if (nodes.count(neighbor_id)) {
                    auto& neighbor_kf = nodes[neighbor_id];

                    std::lock_guard<std::mutex> n_lock(neighbor_kf->mutex);

                    auto& kfs = neighbor_kf->connected_keyframes;
                    auto& ws = neighbor_kf->connected_weights;

                    for (size_t i = 0; i < kfs.size(); ++i)
                        if (kfs[i]->id == id) {
                            kfs.erase(kfs.begin() + i);
                            ws.erase(ws.begin() + i);
                            break;
                        }
                }
            }

        // 2. Remove from central structures
        adjacency_map.erase(id);
        nodes.erase(id);
    }

    std::vector<std::shared_ptr<keyframe>> covisibility_graph::get_connected_keyframes(const std::shared_ptr<keyframe>& kf, int32_t min_weight) const {
        if (!kf)
            return {};

        std::lock_guard<std::mutex> lock(kf->mutex);

        // Faster to read directly from keyframe than querying the map.
        if (min_weight <= 0)
            return kf->connected_keyframes;

        std::vector<std::shared_ptr<keyframe>> result;
        result.reserve(kf->connected_keyframes.size());

        for (size_t i = 0; i < kf->connected_keyframes.size(); ++i)
            if (kf->connected_weights[i] >= min_weight)
                result.push_back(kf->connected_keyframes[i]);

        return result;
    }

    std::vector<std::shared_ptr<keyframe>> covisibility_graph::get_best_covisibility_keyframes(const std::shared_ptr<keyframe>& kf, size_t n) const {
        if (!kf)
            return {};

        std::lock_guard<std::mutex> lock(kf->mutex);

        if (kf->connected_keyframes.empty())
            return {};

        const size_t count = std::min(n, kf->connected_keyframes.size());

        return std::vector<std::shared_ptr<keyframe>>(kf->connected_keyframes.begin(), kf->connected_keyframes.begin() + count);
    }

    std::set<std::shared_ptr<keyframe>> covisibility_graph::get_connected_component(const std::shared_ptr<keyframe>& kf, int32_t depth, int32_t min_weight) const {
        std::set<std::shared_ptr<keyframe>> component;

        if (!kf)
            return component;

        std::queue<std::pair<std::shared_ptr<keyframe>, int>> q;
        q.emplace(kf, 0);
        component.insert(kf);

        while (!q.empty()) {
            auto [current_kf, current_depth] = q.front();
            q.pop();

            if (current_depth >= depth)
                continue;

            // Get neighbors
            std::vector<std::shared_ptr<keyframe>> neighbors = get_connected_keyframes(current_kf, min_weight);

            for (const auto& neighbor : neighbors)
                if (component.find(neighbor) == component.end()) {
                    component.insert(neighbor);
                    q.emplace(neighbor, current_depth + 1);
                }
        }

        return component;
    }

    int32_t covisibility_graph::get_weight(const std::shared_ptr<keyframe>& kf1, const std::shared_ptr<keyframe>& kf2) const {
        if (!kf1 || !kf2)
            return 0;

        std::shared_lock<std::shared_mutex> lock(mutex);

        auto it1 = adjacency_map.find(kf1->id);
        if (it1 != adjacency_map.end()) {
            auto it2 = it1->second.find(kf2->id);
            if (it2 != it1->second.end())
                return it2->second;
        }

        return 0;
    }

    void covisibility_graph::clear() {
        std::unique_lock<std::shared_mutex> lock(mutex);
        adjacency_map.clear();
        nodes.clear();
    }

}   // namespace caai_slam