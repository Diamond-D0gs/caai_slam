#include "caai_slam/utils/euroc_loader.hpp"

#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <iomanip>

namespace caai_slam {

    bool euroc_loader::initialize(const std::string& root_path) {
        dataset_root = root_path;
        camera_idx = 0;
        imu_idx = 0;

        if (!load_camera_data()) {
            fprintf(stderr, "Failed to load camera data from %s\n", root_path.c_str());
            return false;
        }

        if (!load_imu_data()) {
            fprintf(stderr, "Failed to load IMU data from %s\n", root_path.c_str());
            return false;
        }

        load_groundtruth_data();  // Optional, warn if fails

        printf("EuRoC dataset loaded: %zu frames, %zu IMU samples\n", 
               camera_frames.size(), imu_measurements.size());

        return camera_frames.size() > 0 && imu_measurements.size() > 0;
    }

    std::vector<std::string> euroc_loader::split_csv_line(const std::string& line) {
        std::vector<std::string> result;
        std::stringstream ss(line);
        std::string item;
        while (std::getline(ss, item, ',')) {
            result.push_back(item);
        }
        return result;
    }

    bool euroc_loader::load_camera_data() {
        std::string csv_path = dataset_root + "/mav0/cam0/data.csv";
        std::ifstream csv_file(csv_path);
        if (!csv_file.is_open()) {
            fprintf(stderr, "Cannot open: %s\n", csv_path.c_str());
            return false;
        }

        std::string line;
        // Skip header
        std::getline(csv_file, line);

        while (std::getline(csv_file, line)) {
            if (line.empty() || line[0] == '#') continue;

            auto parts = split_csv_line(line);
            if (parts.size() < 2) continue;

            camera_frame frame;
            frame.ts = std::stod(parts[0]) / 1e9;  // Convert nanoseconds to seconds
            frame.image_path = dataset_root + "/mav0/cam0/data/" + parts[1];

            camera_frames.push_back(frame);
        }

        csv_file.close();

        if (camera_frames.empty()) {
            fprintf(stderr, "No camera frames loaded\n");
            return false;
        }

        std::sort(camera_frames.begin(), camera_frames.end(),
                  [](const camera_frame& a, const camera_frame& b) {
                      return a.ts < b.ts;
                  });

        printf("Loaded %zu camera frames\n", camera_frames.size());
        return true;
    }

    bool euroc_loader::load_imu_data() {
        std::string csv_path = dataset_root + "/mav0/imu0/data.csv";
        std::ifstream csv_file(csv_path);
        if (!csv_file.is_open()) {
            fprintf(stderr, "Cannot open: %s\n", csv_path.c_str());
            return false;
        }

        std::string line;
        // Skip header
        std::getline(csv_file, line);

        while (std::getline(csv_file, line)) {
            if (line.empty() || line[0] == '#') continue;

            auto parts = split_csv_line(line);
            if (parts.size() < 7) continue;

            imu_data imu;
            imu.ts = std::stod(parts[0]) / 1e9;  // nanoseconds to seconds
            imu.accel = vec3(std::stod(parts[1]), std::stod(parts[2]), std::stod(parts[3]));
            imu.gyro = vec3(std::stod(parts[4]), std::stod(parts[5]), std::stod(parts[6]));

            imu_measurements.push_back(imu);
        }

        csv_file.close();

        if (imu_measurements.empty()) {
            fprintf(stderr, "No IMU measurements loaded\n");
            return false;
        }

        std::sort(imu_measurements.begin(), imu_measurements.end(),
                  [](const imu_data& a, const imu_data& b) {
                      return a.ts < b.ts;
                  });

        printf("Loaded %zu IMU samples\n", imu_measurements.size());
        return true;
    }

    bool euroc_loader::load_groundtruth_data() {
        std::string csv_path = dataset_root + "/mav0/state_groundtruth_estimate0/data.csv";
        std::ifstream csv_file(csv_path);
        if (!csv_file.is_open()) {
            printf("Note: Ground truth not available at %s\n", csv_path.c_str());
            return false;
        }

        std::string line;
        // Skip header
        std::getline(csv_file, line);

        while (std::getline(csv_file, line)) {
            if (line.empty() || line[0] == '#') continue;

            auto parts = split_csv_line(line);
            if (parts.size() < 8) continue;

            groundtruth_pose pose;
            pose.ts = std::stod(parts[0]) / 1e9;
            pose.position = vec3(std::stod(parts[1]), std::stod(parts[2]), std::stod(parts[3]));
            // Quaternion: x, y, z, w (but EuRoC uses w, x, y, z)
            pose.quaternion = quat(std::stod(parts[7]),    // w
                                   std::stod(parts[4]),    // x
                                   std::stod(parts[5]),    // y
                                   std::stod(parts[6]));   // z

            gt_poses.push_back(pose);
        }

        csv_file.close();

        if (gt_poses.empty()) {
            printf("Note: No ground truth poses loaded\n");
            return false;
        }

        printf("Loaded %zu ground truth poses\n", gt_poses.size());
        return true;
    }

    bool euroc_loader::get_next_camera_frame(camera_frame& out_frame, cv::Mat& out_image) {
        if (camera_idx >= camera_frames.size()) {
            return false;
        }

        out_frame = camera_frames[camera_idx];

        // Load image if not already loaded
        if (!out_frame.is_loaded) {
            out_image = cv::imread(out_frame.image_path, cv::IMREAD_GRAYSCALE);
            if (out_image.empty()) {
                fprintf(stderr, "Failed to load image: %s\n", out_frame.image_path.c_str());
                return false;
            }
            camera_frames[camera_idx].image = out_image;
            camera_frames[camera_idx].is_loaded = true;
        } else {
            out_image = out_frame.image.clone();
        }

        camera_idx++;
        return true;
    }

    void euroc_loader::get_imu_until(const timestamp t_camera, std::vector<imu_measurement>& out_imu_buffer) {
        out_imu_buffer.clear();

        while (imu_idx < imu_measurements.size() && imu_measurements[imu_idx].ts <= t_camera) {
            out_imu_buffer.push_back(imu_measurements[imu_idx]);
            imu_idx++;
        }
    }

    bool euroc_loader::get_next_imu_batch(size_t count, std::vector<imu_measurement>& out_imu) {
        out_imu.clear();

        if (imu_idx >= imu_measurements.size()) {
            return false;
        }

        for (size_t i = 0; i < count && imu_idx < imu_measurements.size(); ++i) {
            out_imu.push_back(imu_measurements[imu_idx]);
            imu_idx++;
        }

        return !out_imu.empty();
    }

    void euroc_loader::reset() {
        camera_idx = 0;
        imu_idx = 0;
    }

    bool euroc_loader::is_finished() const {
        return camera_idx >= camera_frames.size();
    }

    size_t euroc_loader::get_num_frames() const {
        return camera_frames.size();
    }

    size_t euroc_loader::get_current_frame_index() const {
        return camera_idx;
    }

    bool euroc_loader::get_groundtruth_pose(const timestamp ts, se3& out_pose) const {
        if (gt_poses.empty()) {
            return false;
        }

        // Find closest pose by timestamp
        size_t best_idx = 0;
        double min_diff = std::abs(gt_poses[0].ts - ts);

        for (size_t i = 1; i < gt_poses.size(); ++i) {
            double diff = std::abs(gt_poses[i].ts - ts);
            if (diff < min_diff) {
                min_diff = diff;
                best_idx = i;
            }
        }

        const auto& pose = gt_poses[best_idx];
        out_pose.rotation = pose.quaternion;
        out_pose.translation = pose.position;

        return true;
    }

    const std::vector<euroc_loader::groundtruth_pose>& euroc_loader::get_groundtruth_trajectory() const {
        return gt_poses;
    }

} // namespace caai_slam