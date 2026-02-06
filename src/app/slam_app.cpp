#include "caai_slam/app/slam_app.hpp"

#include "caai_slam/mapping/local_map.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <cmath>

namespace caai_slam {
    slam_app::slam_app(const std::string& config_path) : _slam_system(config_path) {
        vis_state.status = system_status::NOT_INITIALIZED;
        vis_state.status_message = "Waiting for dataset...";
    }

    bool slam_app::initialize(const std::string& dataset_root) {
        std::cout << "\n[App] Loading EuRoC dataset from: " << dataset_root << std::endl;

        if (!dataset_loader.initialize(dataset_root)) {
            std::cerr << "[App] Failed to load dataset" << std::endl;
            return false;
        }

        std::cout << "[App] Dataset loaded successfully" << std::endl;
        std::cout << "[App] Total frames: " << dataset_loader.get_num_frames() << std::endl;

        vis_state.status_message = "Dataset loaded. Starting...";
        last_frame_time = std::chrono::high_resolution_clock::now();

        return true;
    }

    bool slam_app::process_next_frame() {
        // 1. Load next camera frame
        euroc_loader::camera_frame cam_frame;
        cv::Mat image;

        if (!dataset_loader.get_next_camera_frame(cam_frame, image)) {
            vis_state.status_message = "Finished processing dataset";
            return false;
        }

        // 2. Get all IMU measurements up to this frame
        dataset_loader.get_imu_until(cam_frame.ts, imu_batch);

        // Start timing
        const auto frame_start = std::chrono::high_resolution_clock::now();

        // 3. Process IMU measurements
        for (const auto& imu : imu_batch) {
            imu_measurement meas;
            meas._timestamp = imu._timestamp;
            meas.linear_acceleration = imu.linear_acceleration;
            meas.angular_velocity = imu.angular_velocity;

            _slam_system.process_imu(meas);
        }

        // 4. Process camera frame
        _slam_system.process_image(image, cam_frame.ts);

        // End timing
        const auto frame_end = std::chrono::high_resolution_clock::now();
        const auto frame_duration = std::chrono::duration<double, std::milli>(frame_end - frame_start);

        // 5. Update performance stats
        perf_stats.total_time_ms = frame_duration.count();
        perf_stats.average_fps = 1000.0 / perf_stats.total_time_ms;

        // 6. Cache visualization state
        {
            vis_state.current_pose = _slam_system.get_current_pose();
            vis_state.status = _slam_system.get_status();

            const auto& local_map = _slam_system.get_local_map();
            vis_state.total_keyframes = local_map->num_keyframes();
            vis_state.total_map_points = local_map->num_map_points();

            // Tracking quality: ratio of matched points to features
            const state cur_state = _slam_system.get_current_state();
            vis_state.tracking_quality = 0.5; // Placeholder; could compute from matched features

            // Status message
            switch (vis_state.status) {
                case system_status::NOT_INITIALIZED:
                    vis_state.status_message = "Initializing...";
                    break;
                case system_status::INITIALIZING:
                    vis_state.status_message = "Detecting gravity and bias...";
                    break;
                case system_status::TRACKING:
                    vis_state.status_message = std::string("Tracking (") +
                        std::to_string(vis_state.total_keyframes) + " KF, " +
                        std::to_string(vis_state.total_map_points) + " MP)";
                    break;
            }
        }

        // 7. Save trajectory entry
        if (vis_state.status == system_status::TRACKING) {
            trajectory_entry entry;
            entry.timestamp = cam_frame.ts;
            entry.pose = _slam_system.get_current_pose();
            entry.velocity = _slam_system.get_current_state().velocity;
            trajectory_history.push_back(entry);
        }

        // Update timing
        last_frame_time = frame_end;

        return true;
    }

    slam_app::visualization_state slam_app::get_vis_state() const {
        return vis_state;
    }

    system_status slam_app::get_status() const {
        return _slam_system.get_status();
    }

    double slam_app::get_progress() const {
        const size_t total = dataset_loader.get_num_frames();
        const size_t current = dataset_loader.get_current_frame_index();
        return (total > 0) ? (static_cast<double>(current) / static_cast<double>(total)) : 0.0;
    }

    bool slam_app::save_trajectory(const std::string& output_path) {
        std::ofstream file(output_path);
        if (!file.is_open()) {
            std::cerr << "[App] Failed to open output file: " << output_path << std::endl;
            return false;
        }

        std::cout << "[App] Saving trajectory to: " << output_path << std::endl;

        // TUM format: timestamp tx ty tz qx qy qz qw
        file << std::fixed << std::setprecision(9);

        for (const auto& entry : trajectory_history) {
            const auto& pose = entry.pose;

            // Convert rotation matrix to quaternion
            const quat q(pose.rotation);

            file << entry.timestamp << " "
                 << pose.translation.x() << " " << pose.translation.y() << " " << pose.translation.z() << " "
                 << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
        }

        file.close();

        std::cout << "[App] Saved " << trajectory_history.size() << " poses" << std::endl;
        return true;
    }

} // namespace caai_slam