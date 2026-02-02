#pragma once

#include "caai_slam/core/types.hpp"

#include <vector>
#include <deque>

namespace caai_slam {
    /**
     * @brief Handles temporal synchronization between camera and IMU sensors
     * 
     * Manages time offsets and provides utilities to extract and interpolate IMU
     * measurements relative to the camera's frame timestamps.
     */
    class time_sync {
    private:
        double offset; // t_imu = t_cam + offset
        double drift_rate; // Time varying drift

    public:
        /**
         * @param initial_offset Initial time offset such that t_imu = t_cam + offset (default 0.0)
         * @param drift_rate Initial clock drift rate (default 0.0)
         */
        explicit time_sync(const double initial_offset = 0.0, const double initial_drift_rate = 0.0) 
            : offset(initial_offset), drift_rate(initial_drift_rate) {}
        
        /**
         * @brief Update the time offset
         * 
         * @param new_offset New offset value in seconds
         */
        inline void set_offset(const double new_offset) { offset = new_offset; }

        /**
         * @brief Update the drift rate
         * 
         * @param new_drift_rate New drift rate value
         */
        inline void set_drift_rate(const double new_drift_rate) { drift_rate = new_drift_rate; }

        /**
         * @brief Convert a camera timestamp to the IMU's time domain
         * 
         * @param t_cam Timestamp in the camera's clock
         * 
         * @return Timestamp in the IMU's clock
         */
        inline double cam_to_imu(const double t_cam) const { return t_cam + offset + (drift_rate * t_cam); } // Linear model

        /**
         * @brief Convert an IMU timestamp to the camera time domain
         * 
         * @param t_imu Timestamp in the IMU's clock
         * 
         * @return Timestamp in the camera's clock
         */
        inline double imu_to_cam(const double t_imu) const { return (t_imu - offset) / (1.0 + drift_rate); }

        /**
         * @brief Extract and interpolate IMU measurements covering a specific camera time interval
         * 
         * This method ensures the returned vector starts exactly at t_cam_start
         * and ends exactly at t_cam_end via linear interpolation of the boundary measurements.
         * 
         * @param t_cam_start Start time of the interval (camera clock)
         * @param t_cam_end End time of the interval (camera clock)
         * @param imu_buffer Buffer of raw IMU measurements
         * 
         * @return Sequence of measurements with the interval [t_cam_start, t_cam_end]
         */
        std::vector<imu_measurement> get_imu_between(const double t_cam_start, const double t_cam_end, const std::deque<imu_measurement>& imu_buffer) const;
    };

} // namespace caai_slam