#include "caai_slam/vio/imu_preintegration.hpp"

#include <cmath>
#include <iostream>

namespace caai_slam {
    imu_preintegration::imu_preintegration(const config& cfg, const imu_bias& initial_bias) {
        // 1. Setup GTSAM preintegration parameters
        // MakeSharedU: gravity points UP in nav frame => n_gravity = [0, 0, -g]
        auto p = gtsam::PreintegratedCombinedMeasurements::Params::MakeSharedU(cfg.imu.gravity_magnitude);

        // =====================================================================
        // Measurement noise covariances (continuous-time densities squared)
        //
        // GTSAM internally scales these by 1/dt during integration, so we
        // provide the continuous-time power spectral density (sigma_c^2).
        // =====================================================================
        const double acc_cov = std::pow(cfg.imu.accel_noise_density, 2);
        const double gyr_cov = std::pow(cfg.imu.gyro_noise_density, 2);

        p->accelerometerCovariance = gtsam::I_3x3 * acc_cov;
        p->gyroscopeCovariance     = gtsam::I_3x3 * gyr_cov;

        // =====================================================================
        // Bias random walk covariances (continuous-time)
        //
        // These define how quickly bias is allowed to drift between keyframes.
        // CombinedImuFactor uses these internally to model the bias evolution
        // constraint between bias_i and bias_j — do NOT add a separate
        // BetweenFactor<imuBias::ConstantBias> or the bias will be double-
        // constrained, causing singular Hessians during marginalization.
        // =====================================================================
        const double acc_rw_cov = std::pow(cfg.imu.accel_random_walk, 2);
        const double gyr_rw_cov = std::pow(cfg.imu.gyro_random_walk, 2);

        p->biasAccCovariance   = gtsam::I_3x3 * acc_rw_cov;
        p->biasOmegaCovariance = gtsam::I_3x3 * gyr_rw_cov;

        // =====================================================================
        // Integration covariance — numerical integration error on position
        //
        // FIX: Previous value of 1e-8 was far too small. This parameter adds
        // a small amount of uncertainty per integration step to account for
        // discretization error. When too small, it makes the preintegration
        // factor excessively informative about the position/velocity block,
        // which can cause ill-conditioning when combined with visual factors.
        //
        // A reasonable value is on the order of the accelerometer noise
        // squared, or slightly larger.
        // =====================================================================
        p->integrationCovariance = gtsam::I_3x3 * 1e-4;

        // =====================================================================
        // Bias-acceleration-omega integration covariance
        //
        // FIX: Previous value of 1e-5 (as I_6x6) was too small and created
        // an extremely tight constraint on how bias interacts with the
        // integrated measurements. This is the covariance of the Jacobian
        // of the preintegrated measurements w.r.t. bias.
        //
        // When this is too small, the CombinedImuFactor's Hessian contribution
        // to the bias block becomes enormous (~1e+11 information), making the
        // system ill-conditioned. After 90 frames of marginalization, these
        // accumulate into a singular frontal Hessian on velocity/bias.
        //
        // Setting this to match the scale of the bias random walk covariances
        // prevents over-constraining.
        // =====================================================================
        p->biasAccOmegaInt = gtsam::I_6x6 * 1e-2;

        // Log parameters for debugging
        std::cout << "[IMU Preintegration] Parameters:" << std::endl;
        std::cout << "  accel_noise_density: " << cfg.imu.accel_noise_density 
                  << " -> cov: " << acc_cov << std::endl;
        std::cout << "  gyro_noise_density:  " << cfg.imu.gyro_noise_density 
                  << " -> cov: " << gyr_cov << std::endl;
        std::cout << "  accel_random_walk:   " << cfg.imu.accel_random_walk 
                  << " -> cov: " << acc_rw_cov << std::endl;
        std::cout << "  gyro_random_walk:    " << cfg.imu.gyro_random_walk 
                  << " -> cov: " << gyr_rw_cov << std::endl;
        std::cout << "  integrationCov:      1e-4" << std::endl;
        std::cout << "  biasAccOmegaInt:     1e-2" << std::endl;

        // 2. Initialize bias
        current_bias_gtsam = gtsam::imuBias::ConstantBias(initial_bias.accelerometer, initial_bias.gyroscope);

        // 3. Initialize the integrator
        preintegrated = std::make_unique<gtsam::PreintegratedCombinedMeasurements>(p, current_bias_gtsam);
    }

    void imu_preintegration::integrate(const imu_measurement& meas, const double dt) {
        if (dt <= 0.0 || dt > 0.5) {
            // FIX: Guard against invalid dt values.
            // dt > 0.5s would indicate a dropped IMU packet or timestamp jump,
            // which would corrupt the preintegration.
            if (dt > 0.5) {
                std::cerr << "[IMU] WARNING: Skipping integration with dt=" << dt 
                          << "s (too large, possible timestamp discontinuity)" << std::endl;
            }
            return;
        }

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

        const gtsam::Pose3 pose_i(start_state.pose.matrix());
        const gtsam::Vector3 vel_i = start_state.velocity;

        const gtsam::NavState state_j = preintegrated->predict(gtsam::NavState(pose_i, vel_i), current_bias_gtsam);

        state result = {};
        result.pose = se3(state_j.pose().rotation().matrix(), state_j.pose().translation());
        result.velocity = state_j.velocity();
        result.bias.accelerometer = current_bias_gtsam.accelerometer();
        result.bias.gyroscope = current_bias_gtsam.gyroscope();
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

    gtsam::PreintegratedCombinedMeasurements imu_preintegration::get_and_reset(const imu_bias& new_bias) {
        std::lock_guard<std::mutex> lock(mutex);

        // FIX: Check that preintegration has actual measurements before returning.
        // An empty preintegration (deltaTij == 0) creates a degenerate IMU factor.
        if (preintegrated->deltaTij() < 1e-6) {
            std::cerr << "[IMU] WARNING: get_and_reset called with near-zero deltaTij ("
                      << preintegrated->deltaTij() << "s). This will create a degenerate "
                      << "CombinedImuFactor." << std::endl;
        }

        const auto result = *preintegrated;

        current_bias_gtsam = gtsam::imuBias::ConstantBias(new_bias.accelerometer, new_bias.gyroscope);
        preintegrated->resetIntegrationAndSetBias(current_bias_gtsam);

        return result;
    }

} // namespace caai_slam
