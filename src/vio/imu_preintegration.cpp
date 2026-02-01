#include "caai_slam/vio/imu_preintegration.hpp"

#include <cmath>

namespace caai_slam {
    imu_preintegration::imu_preintegration(const config& cfg, const imu_bias& initial_bias) {
        // 1. Setup GTSAM preintgration parameters
        // ENU convention usually assumes Z-up, but GTSAM defaults to Z-up for gravity
        auto params = gtsam::PreintegratedCombinedMeasurements::Params::MakeSharedU(cfg.imu.gravity_magnitude);

        // Convert continuous-time noise density to discrete covariance
        // covariance = sigma^2 * I
        const double acc_cov = std::pow(cfg.imu.accel_noise_density, 2);
        const double gyr_cov = std::pow(cfg.imu.gyro_noise_density, 2);
        const double acc_rw_cov = std::pow(cfg.imu.accel_random_walk, 2);
        const double gyr_rw_cov = std::pow(cfg.imu.gyro_random_walk, 2);

        params->accelerometerCovariance = gtsam::I_3x3 * acc_cov;
        params->gyroscopeCovariance = gtsam::I_3x3 * gyr_cov;
        params->integrationCovariance = gtsam::I_3x3 * 1e-8; // Small error integration term

        // Random walk parameters (bias process noise)
        params->biasAccCovariance = gtsam::I_3x3 * acc_rw_cov;
        params->biasOmegaCovariance = gtsam::I_3x3 * gyr_rw_cov;
        params->biasAccOmegaInt = gtsam::I_6x6 * 1e-5; // Bias integration covariance

        // 2. Initialize bias
        current_bias_gtsam = gtsam::imuBias::ConstantBias(initial_bias.accelerometer, initial_bias.gyroscope);

        // 3. Initialize the integrator
        preintegrated = std::make_unique<gtsam::PreintegratedCombinedMeasurements>(params, current_bias_gtsam);
    }

    void imu_preintegration::integrate(const imu_measurement& meas, const double dt) {
        std::lock_guard<std::mutex> lock(mutex);
        preintegrated->integrateMeasurement(meas.linear_acceleration, meas.angular_velocity, dt);
    }

    void imu_preintegration::reset(const imu_bias& new_bias) {
        std::lock_guard<std::mutex> lock(mutex);
        current_bias_gtsam = gtsam::imuBias::ConstantBias(new_bias.accelerometer, new_bias.gyroscope);
        preintegrated->resetIntegrationAndSetBias(current_bias_gtsam);
    }

    state imu_preintegration::predict(const state& start_state) const {
        std::lock_guard<std::mutex> lock(mutex);

        // Convert start state to GTSAM types
        const gtsam::Pose3 pose_i(start_state.pose.matrix());
        const gtsam::Vector3 vel_i = start_state.velocity;

        // Predict
        const gtsam::NavState state_j = preintegrated->predict(gtsam::NavState(pose_i, vel_i), current_bias_gtsam);

        // Convert back to custom types
        state result = {};
        result.pose = se3(state_j.pose().rotation().matrix(), state_j.pose().translation());
        result.velocity = state_j.velocity();

        // Bias remains constant during prediction step (random walk happens in graph optimization)
        result.bias.accelerometer = current_bias_gtsam.accelerometer();
        result.bias.gyroscope = current_bias_gtsam.gyroscope();

        // Propagate timestamp
        result._timestamp = start_state._timestamp + preintegrated->deltaTij();

        return result;
    }

    gtsam::PreintegratedCombinedMeasurements imu_preintegration::get_preintegrated() const {
        std::lock_guard<std::mutex> lock(mutex);
        return *preintegrated;
    }

    imu_bias imu_preintegration::get_current_bias() const {
        std::lock_guard<std::mutex> lock(mutex);

        imu_bias b = {};
        b.accelerometer = current_bias_gtsam.accelerometer();
        b.gyroscope = current_bias_gtsam.gyroscope();

        return b;
    }

} // namespace caai_slam