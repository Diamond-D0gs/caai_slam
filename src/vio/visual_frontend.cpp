#include "caai_slam/vio/visual_frontend.hpp"

#include <opencv2/calib3d.hpp>

namespace caai_slam {
    visual_frontend::visual_frontend(const config& cfg, std::shared_ptr<local_map> l_map) : _config(cfg), _local_map(l_map) {
        extractor = std::make_unique<feature_extractor>(cfg);
        matcher = std::make_unique<feature_matcher>(cfg);
    }

    uint32_t visual_frontend::track_last_frame(std::shared_ptr<frame>& curr_frame, const std::shared_ptr<frame>& prev_frame) {
        const std::vector<cv::DMatch> matches = matcher->match(curr_frame->descriptors, prev_frame->descriptors);

        uint32_t count = 0;
        for (const auto& m : matches)
            if (prev_frame->has_map_point(m.trainIdx)) {
                curr_frame->map_points[m.queryIdx] = prev_frame->map_points[m.trainIdx];
                ++count;
            }

        return count;
    }

    uint32_t visual_frontend::track_local_map(std::shared_ptr<frame>& curr_frame) {
        // Project map points into the camera using the predicted pose.
        const se3 t_world_cam = curr_frame->pose * _config._extrinsics.t_cam_imu.inverse();
        const auto candidates = _local_map->get_map_points_in_view(t_world_cam);

        uint32_t count = 0;
        for (const auto& mp : candidates)
            if (mp && !mp->is_bad) {
                // Simple search: match mp descriptor against current frame descriptors
                // TODO: Consider upgrading to a spatial search grid for higher perf.
                int32_t best_idx = -1;
                double best_dist = 50.0;
                for (size_t i = 0; i < curr_frame->keypoints.size(); ++i)
                    if (!curr_frame->map_points[i]) {
                        const double dist = cv::norm(curr_frame->descriptors.row(i), mp->descriptor, cv::NORM_HAMMING);
                        if (dist < best_dist) {
                            best_idx = static_cast<int32_t>(i);
                            best_dist = dist;
                        }
                    }

                if (best_idx >= 0) {
                    curr_frame->map_points[best_idx] = mp;
                    ++count;
                }
            }

        return count;
    }

    void visual_frontend::outlier_rejection(std::shared_ptr<frame>& curr_frame) {
        // Collect 2D-3D correspondences
        std::vector<cv::Point3f> pts_3d;
        std::vector<cv::Point2f> pts_2d;
        std::vector<size_t> indices;

        for (size_t i = 0; i < curr_frame->map_points.size(); ++i)
            if (curr_frame->has_map_point(i)) {
                const auto& p = curr_frame->map_points[i]->position;
                pts_2d.push_back(curr_frame->keypoints[i].pt);
                pts_3d.emplace_back(p.x(), p.y(), p.z());
                indices.push_back(i);
            }

        if (pts_3d.size() < 4)
            return;

        // Use RANSAC PnP to find inliers
        const cv::Mat k = (cv::Mat_<double>(3,3) << _config.camera.fx, 0, _config.camera.cx, 0, _config.camera.fy, _config.camera.cy, 0, 0, 1);
        const cv::Mat dist_coeffs = (cv::Mat_<double>(4, 1) << _config.camera.k1, _config.camera.k2, _config.camera.p1, _config.camera.p2);

        cv::Mat rvec, tvec;
        std::vector<int32_t> inliers;
        cv::solvePnPRansac(pts_3d, pts_2d, k, dist_coeffs, rvec, tvec, false, 100, 2.0, 0.99, inliers);

        // Convert inliers to set for fast lookup
        std::unordered_set<int32_t> inlier_set(inliers.begin(), inliers.end());

        // Nullify outliers
        for (size_t i = 0; i < indices.size(); ++i)
            if (inlier_set.find(static_cast<int32_t>(i)) == inlier_set.end())
                curr_frame->map_points[indices[i]] = nullptr;
    }

    std::shared_ptr<frame> visual_frontend::process_image(const cv::Mat& image, const timestamp ts, const se3& predicted_pose) {
        std::lock_guard<std::mutex> lock(mutex);

        // 1. Feature extraction
        cv::Mat descs;
        std::vector<cv::KeyPoint> kps;
        extractor->detect_and_compute(image, kps, descs);

        auto curr_frame = std::make_shared<frame>(ts, kps, descs);
        curr_frame->pose = predicted_pose;

        // 2. Tracking
        const uint32_t matches_prev = (last_frame) ? track_last_frame(curr_frame, last_frame) : 0;
        const uint32_t matches_map = track_local_map(curr_frame);

        // 3. Geometry validation
        if (matches_prev + matches_map > 10)
            outlier_rejection(curr_frame);

        last_frame = curr_frame;

        return curr_frame;
    }

    bool visual_frontend::need_new_keyframe(const std::shared_ptr<frame>& curr_frame, const std::shared_ptr<keyframe>& last_kf) {
        if (!curr_frame || !last_kf)
            return true;

        // 1. Count tracked map points
        uint32_t tracked_mps = 0;
        for (const auto& mp : curr_frame->map_points)
            if (mp && !mp->is_bad)
                ++tracked_mps;

        // 2. Decision logic
        // Keyframe if tracking weakening
        if (tracked_mps < static_cast<uint32_t>(_config.frontend.min_matches_tracking) * 2)
            return true;

        // Keyframe if enough time has passed
        if ((curr_frame->_timestamp - last_kf->_timestamp) > 0.5)
            return true;

        // Keyframe if there has been significant movement (parallax/displacement)
        if (last_kf->get_pose().between(gtsam::Pose3(curr_frame->pose.matrix())).translation().norm() > 0.3)
            return true;

        return false;
    }

} // namespace caai_slam