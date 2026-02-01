#include "caai_slam/frontend/frame.hpp"

namespace caai_slam {
    std::atomic<uint64_t> frame::next_id{0};

    frame::frame(timestamp ts, const std::vector<cv::KeyPoint>& kps, const cv::Mat& descs) : id(next_id++), _timestamp(ts), keypoints(kps) {
        descs.copyTo(descriptors);

        // Initialize map point associations as nullptr
        map_points.resize(keypoints.size(), nullptr);

        // Initialize default state
        velocity = vec3::Zero();
        bias = imu_bias();
    }

    bool frame::has_map_point(const size_t idx) const {
        return (idx >= map_points.size()) ? false : (map_points[idx] && !map_points[idx]->is_bad);
    }

    vec3 frame::get_camera_center(const se3& t_cam_imu) const {
        return (pose * t_cam_imu.inverse()).translation;
    }

} // namespace caai_slam