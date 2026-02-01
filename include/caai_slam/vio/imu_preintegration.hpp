#pragma once

#include "caai_slam/core/config.hpp"
#include "caai_slam/core/types.hpp"

#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/ImuFactor.h>

#include <memory>
#include <mutex>

namespace caai_slam {
    /**
     * @brief Handles high-frequency IMU integration between keyframes
     * 
     * Uses GTSAM's PreintegratedCombinedMeasurements to accumulate
     * IMU readings into a relative motion constraint (delta pose, velocity, bias)
     */
    class imu_preintegration {
    private:
        // GTSAM IMU parameters (noise models, gravity)
        boost::shared_ptr<gtsam::PreintegratedCombinedMeasurements::Params> params; // Boost shared_ptr for legacy purposes

        // Integrator
        std::unique_ptr<gtsam::PreintegratedCombinedMeasurements> preintegrated;

        // Current bias estimate
        gtsam::imuBias::ConstantBias current_bias_gtsam;

        // Thread safety
        mutable std::mutex mutex;

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        /**
         * @brief Construct a new IMU preintegration object
         * 
         * @param cfg System configuration containing IMU noise parameters
         * @param initial_bias Initial estimate of accelerometer and gyroscope biases
         */
        explicit imu_preintegration(const config& cfg, const imu_bias& initial_bias = imu_bias());

        /**
         * @brief Integrate a single IMU measurement
         * 
         * @param meas The IMU measurement (accel, gyro, timestamp)
         * @param dt Time deleta since the last measurement in seconds
         */
        void integrate(const imu_measurement& meas, const double dt);

        /**
         * @brief Reset integration with a new bias estimate
         * 
         * Typicaly called after optimization updates the bias for a keyframe
         * 
         * @param new_bias The updates bias estimate
         */
        void reset(const imu_bias& new_bias);

        /**
         * @brief Predict the state at the current time using the integrated measurements
         * 
         * Used for state propagation (e.g., to initialize a new keyframe pose)
         * 
         * @param start_state The state at the beginning of the integration interval
         * 
         * @return The predicted state at the end of the interval
         */
        state predict(const state& start_state) const;

        /**
         * @brief Get the internal GTSAM preintegration object
         * 
         * @return The accumulated measurements
         */
        gtsam::PreintegratedCombinedMeasurements get_preintegrated() const;

        /**
         * @brief Get the current bias estimate used for integration
         * 
         * @return Current bias
         */
        imu_bias get_current_bias() const;
    };

} // namespace caai_slam