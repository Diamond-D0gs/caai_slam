#pragma once

#include "caai_slam/frontend/frame.hpp"
#include "caai_slam/core/config.hpp"
#include "caai_slam/core/types.hpp"

#include <memory>
#include <atomic>
#include <vector>
#include <mutex>

namespace caai_slam {
    // Forward declariations
    class fixed_lag_smoother;
    class imu_preintegration;
    class keyframe_database;
    class vio_initializer;
    class visual_frontend;
    class graph_optimizer;
    class feature_matcher;
    class loop_detector;
    class local_map;
    class time_sync;

    /**
     * @brief System status flags
     */
    enum class system_status {
        NOT_INITIALIZED, // Waiting for data/initialization
        INITIALIZING, // Accumulating data for static initialization
        TRACKING, // Normal operation
        LOST // Tracking failure (not fully implemented)
    };

    /**
     * @brief Main entry point for the CAAI-SLAM system
     * 
     * Orchestates the interaction between:
     * - Visual frontend (tracking)
     * - Backend (fixed-lag smoothing)
     * - VIO initialization
     * - Loop closure
     * - Mapping
     */
    class slam_system {
    private:
        config _config;

        // System state
        std::atomic<system_status> status;
        state current_state;

        // Modules
        std::unique_ptr<graph_optimizer> _loop_closure_optimizer;
        std::unique_ptr<fixed_lag_smoother> _fixed_lag_smoother;
        std::unique_ptr<imu_preintegration> _imu_preintegration;
        std::shared_ptr<keyframe_database> _keyframe_database;
        std::unique_ptr<visual_frontend> _visual_frontend;
        std::unique_ptr<vio_initializer> _vio_initializer;
        std::unique_ptr<feature_matcher> _feature_matcher;
        std::unique_ptr<loop_detector> _loop_detector;
        std::shared_ptr<local_map> _local_map;
        std::unique_ptr<time_sync> _time_sync;

        // Runtime tracking
        std::shared_ptr<keyframe> last_keyframe;
        double last_image_timestamp = -1.0;
        double last_imu_time = -1.0;

        // Thread safety
        mutable std::mutex mutex;

        /**
         * @brief Handle the initialization phase
         * 
         * @param image Current image
         * @param ts Current timestamp
         */
        void process_initialization(const cv::Mat& image, const double ts);

        /**
         * @brief Handle the tracking phase
         * 
         * @param image Current image
         * @param ts Current timestamp
         */
        void process_tracking(const cv::Mat& image, const double ts);

        /**
         * @brief Create a new keyframe from the currentframe and insert it into the backend
         * 
         * @param curr_frame the current tracked frame
         */
        void create_and_insert_keyframe(const std::shared_ptr<frame>& curr_frame);

        void correct_loop_closure(std::shared_ptr<keyframe>& curr_kf);

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        /**
         * @param cfg_path Path to the YAML configuration file
         */
        explicit slam_system(const std::string& cfg_path);

        // Disable copying
        slam_system(const slam_system&) = delete;
        slam_system& operator=(const slam_system&) = delete;

        /**
         * @brief Process a raw IMU measurement
         * 
         * @param meas The IMU data (accel, gyro, timestamp)
         */
        void process_imu(const imu_measurement& meas);

        /**
         * @brief Process a raw camera image
         * 
         * @param image Grayscale image data
         * @param ts Image acquisition timestamp (seconds)
         */
        void process_image(const cv::Mat& image, const double ts);

        /**
         * @brief Get the current estimated pose of the IMU in the world frame
         * 
         * @return t_world_imu
         */
        se3 get_current_pose() const;

        /**
         * @brief Get the full current state (pose, velocity, bias)
         * 
         * @return Current system state
         */
        state get_current_state() const;

        /**
         * @brief Get the current system status
         * 
         * @return The system status
         */
        system_status get_status() const;

        /**
         * @brief Reset the system to a clean state
         */
        void reset();
    };

} // namespace caai_slam