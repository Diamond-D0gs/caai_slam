#pragma once

#include "caai_slam/frontend/frame.hpp"
#include "caai_slam/core/config.hpp"
#include "caai_slam/core/types.hpp"

#include <memory>
#include <vector>
#include <mutex>
#include <deque>

namespace caai_slam {
    /**
     * @brief Initialization status of the VIO system
     */
    enum class initialization_status {
        NOT_READY, // Insufficient data to attempt initialization
        SUCCESS, // Initialization successful, state estimated
        FAILED // Initialization failed due to poor data quality or motion
    };

    /**
     * @brief Handles the initial estimation of the SLAM state
     * 
     * Performs static alignment to estimate gravity and initial orientation.
     * Buffers frames and IMU data to ensure a clean start for the filter/graph.
     */
    class vio_initializer {
    private:
        static constexpr uint32_t FRAME_BUFFER_MAX_SIZE = 20U;
        static constexpr double STATIC_THRESHOLD = 0.05;

        config _config;

        std::deque<imu_measurement> imu_buffer;
        std::deque<std::shared_ptr<frame>> frame_buffer;

        // Thread safety
        mutable std::mutex mutex;

        /**
         * @brief Check if the IMU data indicates the device is stationary
         * 
         * Uses the variance of the acceleration magnitude over the buffered window.
         * Calling method must be holding mutex.
         * 
         * @return True if the variance is below the static threshold
         */
        bool is_static() const;

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        /**
         * @param cfg System configuration containing IMU noise parameters
         */
        explicit vio_initializer(const config& cfg) : _config(cfg) {}

        /**
         * @brief Attempt to estimate the initial state (static initialization)
         * 
         * Logic: Analyzes buffered IMU data to ensure the device is static,
         * calculates average acceleration to find gravity direction, and aligns
         * the world frame such that gravity points downwards (-Z).
         * 
         * @param out_initial_state Output state containing the estimated pose, velocity, and biases
         * 
         * @return Status of the attemp
         */
        initialization_status try_initialize(state& out_initial_state);

        /**
         * @brief Add IMU measurement to initialization buffer
         * 
         * @param meas IMU measurement data
         */
        void add_imu(const imu_measurement& meas);

        /**
         * @brief Add a visual frame to the initialization buffer
         * 
         * @param f Shared pointer to the processed frame
         */
        void add_frame(const std::shared_ptr<frame>& f);

        /**
         * @brief Clear all buffered data to start initialization over
         */
        void reset();
    };

} // namespace caai_slam