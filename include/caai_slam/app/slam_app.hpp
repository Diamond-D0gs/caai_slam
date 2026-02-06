#pragma once

#include "caai_slam/utils/euroc_loader.hpp"
#include "caai_slam/core/slam_system.hpp"

#include <opencv2/core.hpp>

#include <vector>
#include <memory>
#include <chrono>

namespace caai_slam {
    /**
     * @brief Common SLAM application framework
     * 
     * Handles:
     * - Dataset loading (EuRoC format)
     * - SLAM processing loop
     * - Trajectory logging
     * - State caching for visualization
     * - Performance statistics
     */
    class slam_app {
    public:
        struct visualization_state {
            se3 current_pose;
            system_status status;
            size_t total_keyframes;
            size_t total_map_points;
            double tracking_quality; // 0.0 to 1.0
            std::string status_message;
        };

        struct performance_stats {
            double frontend_time_ms;
            double backend_time_ms;
            double total_time_ms;
            double average_fps;
        };   

    protected:
        struct trajectory_entry {
            double timestamp;
            vec3 velocity;
            se3 pose;
        };

        euroc_loader dataset_loader;
        slam_system _slam_system;

        // Trajectory history for saving
        std::vector<trajectory_entry> trajectory_history;

        // Performance tracking
        performance_stats perf_stats = {};

        // Visualization state
        mutable visualization_state vis_state = {};

        // Timing
        std::chrono::high_resolution_clock::time_point last_frame_time;

        // IMU buffer for current frame
        std::vector<imu_measurement> imu_batch;

    public:
        slam_app(const std::string& config_path);
        virtual ~slam_app() = default;

        /**
         * @brief Initialize the application with a dataset
         * 
         * @param dataset_root Root directory of EuRoC dataset
         * 
         * @return Success flag
         */
        bool initialize(const std::string& dataset_root);

        /**
         * @brief Process one frame (called by main loop)
         * 
         * @return True if more frames available, False if dataset exhausted
         */
        bool process_next_frame();

        /**
         * @brief Get current visualization state
         */
        visualization_state get_vis_state() const;

        /**
         * @brief Get performance metrics
         */
        performance_stats get_perf_stats() const { return perf_stats; }

        /**
         * @brief Get current SLAM system status
         */
        system_status get_status() const;

        /**
         * @brief Check if processing is complete
         */
        inline bool is_finished() const { return dataset_loader.is_finished(); }

        /**
         * @brief Get progress ratio [0, 1]
         */
        double get_progress() const;

        /**
         * @brief Save trajectory to file (TUM format)
         */
        bool save_trajectory(const std::string& output_path);

        /**
         * @brief Get total frames in dataset
         */
        inline size_t get_total_frames() const { return dataset_loader.get_num_frames(); }

        /**
         * @brief Get current frame index
         */
        inline size_t get_current_frame() const { return dataset_loader.get_current_frame_index(); }

        /**
         * @brief Reset SLAM system
         */
        inline void reset_slam() { _slam_system.reset(); }
    };

} // namespace caai_slam