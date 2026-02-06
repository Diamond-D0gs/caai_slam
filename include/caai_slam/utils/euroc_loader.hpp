#pragma once

#include "caai_slam/core/types.hpp"
#include "caai_slam/core/config.hpp"

#include <string>
#include <vector>
#include <memory>
#include <fstream>
#include <sstream>

namespace caai_slam {

    /**
     * @brief Loads and manages EuRoC dataset playback
     * 
     * EuRoC dataset structure:
     * dataset/
     * ├── mav0/
     * │   ├── cam0/
     * │   │   ├── data/        (images)
     * │   │   └── data.csv     (timestamps)
     * │   ├── imu0/
     * │   │   └── data.csv     (imu measurements)
     * │   └── state_groundtruth_estimate0/
     * │       └── data.csv     (ground truth poses)
     * └── ...
     */
    class euroc_loader {
    public:
        struct camera_frame {
            timestamp ts;
            std::string image_path;
            cv::Mat image;  // Loaded image
            bool is_loaded = false;
        };

        struct imu_data {
            timestamp ts;
            vec3 accel;
            vec3 gyro;
        };

        struct groundtruth_pose {
            timestamp ts;
            vec3 position;
            quat quaternion;  // w, x, y, z
        };

    private:
        std::string dataset_root;
        std::vector<camera_frame> camera_frames;
        std::vector<imu_data> imu_measurements;
        std::vector<groundtruth_pose> gt_poses;

        size_t camera_idx = 0;
        size_t imu_idx = 0;

        // Helper functions
        bool load_camera_data();
        bool load_imu_data();
        bool load_groundtruth_data();
        
        std::vector<std::string> split_csv_line(const std::string& line);

    public:
        /**
         * @brief Initialize the EuRoC dataset loader
         * 
         * @param root_path Path to the dataset root directory
         * @return True if dataset loaded successfully
         */
        bool initialize(const std::string& root_path);

        /**
         * @brief Get the next camera frame
         * 
         * @param out_frame Output frame data
         * @param out_image Output image (grayscale)
         * @return True if a frame was available, false if end of dataset
         */
        bool get_next_camera_frame(camera_frame& out_frame, cv::Mat& out_image);

        /**
         * @brief Get all IMU measurements up to a camera timestamp
         * 
         * @param t_camera Camera timestamp
         * @param out_imu_buffer Output vector of IMU measurements up to t_camera
         */
        void get_imu_until(const timestamp t_camera, std::vector<imu_measurement>& out_imu_buffer);

        /**
         * @brief Get the next batch of IMU measurements
         * 
         * @param count Number of measurements to retrieve
         * @param out_imu Output vector
         * @return True if measurements were available
         */
        bool get_next_imu_batch(size_t count, std::vector<imu_measurement>& out_imu);

        /**
         * @brief Reset to the beginning of the dataset
         */
        void reset();

        /**
         * @brief Check if we've reached the end
         * 
         * @return True if no more camera frames
         */
        bool is_finished() const;

        /**
         * @brief Get total number of frames in dataset
         */
        size_t get_num_frames() const;

        /**
         * @brief Get current frame index
         */
        size_t get_current_frame_index() const;

        /**
         * @brief Get groundtruth pose at specified timestamp (linear interpolation)
         * 
         * @param ts Timestamp
         * @param out_pose Output pose
         * @return True if pose found
         */
        bool get_groundtruth_pose(const timestamp ts, se3& out_pose) const;

        /**
         * @brief Get groundtruth trajectory
         */
        const std::vector<groundtruth_pose>& get_groundtruth_trajectory() const;
    };

} // namespace caai_slam