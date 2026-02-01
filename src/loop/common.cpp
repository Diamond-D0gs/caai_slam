#include "caai_slam/loop/common.hpp"

#include <opencv2/features2d.hpp>

namespace caai_slam {
    void get_matched_points(const float match_ratio_thresh, const std::shared_ptr<keyframe>& query, const std::shared_ptr<keyframe>& candidate, Eigen::Matrix<double, 3, Eigen::Dynamic>& src_cloud, Eigen::Matrix<double, 3, Eigen::Dynamic>& target_cloud) {
        std::vector<std::vector<cv::DMatch>> knn_matches;
        const cv::BFMatcher matcher(cv::NORM_HAMMING);
        matcher.knnMatch(query->descriptors, candidate->descriptors, knn_matches, 2);

        std::vector<vec3> pts_query, pts_cand;
        pts_query.reserve(knn_matches.size());
        pts_cand.reserve(knn_matches.size());

        const gtsam::Pose3 temp_query_w = query->get_pose().inverse(); 
        const gtsam::Pose3 temp_cand_w = candidate->get_pose().inverse(); 

        const se3 t_query_w(temp_query_w.rotation().matrix(), temp_query_w.translation()); // World to query cam
        const se3 t_cand_w(temp_cand_w.rotation().matrix(), temp_cand_w.translation()); // World to cand cam

        for (const auto& m : knn_matches)
            if (m.size() >= 2 && m[0].distance < match_ratio_thresh * m[1].distance) {
                const auto& mp_c = candidate->map_points[m[0].trainIdx];
                const auto& mp_q = query->map_points[m[0].queryIdx];
                if (mp_q && mp_c && !mp_q->is_bad && !mp_c->is_bad) {
                    // Lift to local IMU frame : p_imu = t_imu_world * p_world
                    pts_query.emplace_back(t_query_w * mp_q->position);
                    pts_cand.emplace_back(t_cand_w * mp_c->position);
                }
            }

        // Fill Eigen matrices
        src_cloud.resize(3, pts_query.size());
        target_cloud.resize(3, pts_cand.size());

        for (size_t i = 0; i < pts_query.size(); ++i) {
            src_cloud.col(i) = pts_query[i];
            target_cloud.col(i) = pts_cand[i];
        }
    }

} // namespace caai_slam