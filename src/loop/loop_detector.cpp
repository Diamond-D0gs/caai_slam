#include "caai_slam/loop/loop_detector.hpp"
#include "caai_slam/loop/common.hpp"

#include <opencv2/features2d.hpp>

#include <algorithm>
#include <iostream>
#include <map>

namespace caai_slam {
    loop_detector::loop_detector(const config& cfg) : _config(cfg) {
        vocab = std::make_unique<fbow::Vocabulary>();

        // Configure TEASER++
        // Default parameters are usually robust, but we tune for SLAM
        teaser::RobustRegistrationSolver::Params params = {};
        params.noise_bound = cfg.backend.loop_closure_noise_pos; // e.g. 0.05m
        params.rotation_cost_threshold = 1e-6;
        params.rotation_max_iterations = 100;
        params.rotation_gnc_factor = 1.4;
        params.estimate_scaling = false; // VIO is metric, scale is 1.0
        params.cbar2 = 1.0;
        
        teaser_solver = std::make_unique<teaser::RobustRegistrationSolver>(params); 
    }

    std::vector<std::shared_ptr<keyframe>> loop_detector::query_database(const std::shared_ptr<keyframe>& kf, const std::vector<std::shared_ptr<keyframe>>& active_kfs) {
        std::lock_guard<std::mutex> lock(mutex);

        if (database_keyframes.empty())
            return {};

        std::unordered_set<uint64_t> active_ids;
        for (const auto& active : active_kfs)
            active_ids.insert(active->id);

        // 1. Accumulate scores from inverted index
        // To speed up, only words that exist in the query frame are checked
        std::unordered_map<keyframe*, double> scores;
        for (const auto& [word_id, word_val] : kf->bow_vec)
            if (word_id < inverted_index.size())
                for (const auto& db_kf : inverted_index[word_id]) {
                    // Exclusion 1. Ignore active keyframes (too close, usually handled by tracking)
                    if (active_ids.count(db_kf->id))
                        continue;

                    // Exclusion 2. Ignore recent neighbors by ID (temporal exclusion)
                    // Prevents closing loop with immediate predecessor
                    if (kf->id > db_kf->id && (kf->id - db_kf->id) < static_cast<uint64_t>(_config.loop.exclude_recent_n))
                        continue;

                    scores[db_kf.get()] += word_val * db_kf->bow_vec.at(word_id); // Dot product approximation
                }

        // 2. Filter & sort
        std::vector<std::pair<double, std::shared_ptr<keyframe>>> sorted_candidates;
        for (const auto& [kf_ptr, score] : scores)
            // Normalized score: score / (norm(a) * norm(b))
            // FBoW vectors are usually L2 normalized, so the dot product is the cosine similiarity.
            if (kf_ptr && score > _config.loop.similiarity_threshold)
                sorted_candidates.emplace_back(score, kf_ptr->shared_from_this());
        
        std::sort(sorted_candidates.begin(), sorted_candidates.end(), [](const auto& a, const auto& b) { return a.first > b.first; });

        // Return top 3 candidates
        std::vector<std::shared_ptr<keyframe>> result;
        
        size_t count = 0;
        for (const auto& [score, canidate_kf] : sorted_candidates) {
            result.push_back(canidate_kf);
            if (++count >= 3)
                break;
        }

        return result;
    }

    bool loop_detector::verify_geometry(const std::shared_ptr<keyframe>& query, const std::shared_ptr<keyframe>& candidate, se3& out_t_cand_query, uint32_t& out_inliers) {
        // 1. Prepare 3D point clouds for TEASER++
        Eigen::Matrix<double, 3, Eigen::Dynamic> src_cloud, target_cloud; // Query and candidate frame
        caai_slam::get_matched_points(_config.frontend.match_ratio_thresh, query, candidate, src_cloud, target_cloud);

        if (src_cloud.cols() < _config.loop.min_matches_geom)
            return false;

        // 2. Solve with TEASER++
        // We want t_cand_query (aligns query cloud to candidate cloud)
        // target = t * src => candidate_points = t * query_points
        teaser_solver->solve(src_cloud, target_cloud);

        const auto solution = teaser_solver->getSolution();

        if (!solution.valid)
            return false;

        // 3. Validate inliers
        const uint32_t valid_inliers = count_inliers(_config.backend.loop_closure_noise_pos, solution, src_cloud, target_cloud);
        if (valid_inliers < static_cast<uint32_t>(_config.loop.min_matches_geom))
            return false;

        out_inliers = valid_inliers;
        out_t_cand_query = se3(solution.rotation, solution.translation);

        return true;
    }

    // void loop_detector::get_matched_points(const std::shared_ptr<keyframe>& query, const std::shared_ptr<keyframe>& candidate, Eigen::Matrix<double, 3, Eigen::Dynamic>& src_cloud, Eigen::Matrix<double, 3, Eigen::Dynamic>& target_cloud) {
    //     std::vector<std::vector<cv::DMatch>> knn_matches;
    //     const cv::BFMatcher matcher(cv::NORM_HAMMING);
    //     matcher.knnMatch(query->descriptors, candidate->descriptors, knn_matches, 2);
    //
    //     std::vector<vec3> pts_query, pts_cand;
    //     pts_query.reserve(knn_matches.size());
    //     pts_cand.reserve(knn_matches.size());
    //
    //     const gtsam::Pose3 temp_query_w = query->get_pose().inverse(); 
    //     const gtsam::Pose3 temp_cand_w = candidate->get_pose().inverse(); 
    //
    //     const se3 t_query_w(temp_query_w.rotation().matrix(), temp_query_w.translation()); // World to query cam
    //     const se3 t_cand_w(temp_cand_w.rotation().matrix(), temp_cand_w.translation()); // World to cand cam
    //
    //     for (const auto& m : knn_matches)
    //         if (m.size() >= 2 && m[0].distance < _config.frontend.match_ratio_thresh * m[1].distance) {
    //             const auto& mp_c = candidate->map_points[m[0].trainIdx];
    //             const auto& mp_q = query->map_points[m[0].queryIdx];
    //             if (mp_q && mp_c && !mp_q->is_bad && !mp_c->is_bad) {
    //                 // Lift to local IMU frame : p_imu = t_imu_world * p_world
    //                 pts_query.emplace_back(t_query_w * mp_q->position);
    //                 pts_cand.emplace_back(t_cand_w * mp_c->position);
    //             }
    //         }
    //
    //     // Fill Eigen matrices
    //     src_cloud.resize(3, pts_query.size());
    //     target_cloud.resize(3, pts_cand.size());
    //
    //     for (size_t i = 0; i < pts_query.size(); ++i) {
    //         src_cloud.col(i) = pts_query[i];
    //         target_cloud.col(i) = pts_cand[i];
    //     }
    // }
    //
    // void loop_detector::reset() {
    //     std::lock_guard<std::mutex> lock(mutex);
    //
    //     inverted_index.clear();
    //     if (vocab->isValid())
    //         inverted_index.resize(vocab->size());
    //
    //     database_keyframes.clear();
    // }

    bool loop_detector::load_vocabulary(const std::string& path) {
        std::lock_guard<std::mutex> lock(mutex);

        try {
            vocab->readFromFile(path);
            if (!vocab->isValid()) {
                std::cerr << "LoopDetector: Invalid vocabulary file: " << path << std::endl;
                return false;
            }

            inverted_index.resize(vocab->size());
        }
        catch (const std::exception& e) {
            std::cerr << "LoopDetector: Failed to load vocabulary: " << e.what() << std::endl;
            return false;
        }

        return true;
    }

    void loop_detector::add_keyframe(const std::shared_ptr<keyframe>& kf) {
        if (!kf || !vocab->isValid())
            return;

        std::lock_guard<std::mutex> lock(mutex);

        // Compute BoW if not already present
        if (kf->bow_vec.empty())
            kf->compute_bow(*vocab);

        database_keyframes.push_back(kf);

        // Update inverted index
        for (const auto& [word_id, unused] : kf->bow_vec)
            if (word_id < inverted_index.size())
                inverted_index[word_id].push_back(kf);
    }

    loop_detector::loop_result loop_detector::detect_loop(const std::shared_ptr<keyframe>& kf, const std::vector<std::shared_ptr<keyframe>>& active_kfs) {
        if (!_config.loop.enable || !vocab->isValid())
            return {};

        // 1. Find candidates via FBoW
        std::vector<std::shared_ptr<keyframe>> candidates = query_database(kf, active_kfs);
        if (candidates.empty())
            return {};

        // 2. Geometric verification (TEASER++)
        for (const auto& candidate : candidates) {
            se3 t_cand_query;
            uint32_t inliers = 0;
            if (verify_geometry(kf, candidate, t_cand_query, inliers)) {
                loop_result result = {};
                result.inliers = static_cast<int32_t>(inliers);
                result.t_match_query = t_cand_query;
                result.match_kf = candidate;
                result.is_detected = true;
                result.query_kf = kf;

                std::cout << "Loop Detected: KF " << kf->id << " matches KF " << candidate->id << " with " << inliers << " inliers." << std::endl;

                return result;
            }
        }

        return {};
    }

} // namespace caai_slam