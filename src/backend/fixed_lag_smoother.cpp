#include "caai_slam/backend/fixed_lag_smoother.hpp"

#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/inference/Symbol.h>

namespace caai_slam {
    // Helper symbol generators
    inline gtsam::Symbol sym_pose(uint64_t id) { return gtsam::Symbol('x', id); }
    inline gtsam::Symbol sym_vel(uint64_t id) { return gtsam::Symbol('v', id); }
    inline gtsam::Symbol sym_bias(uint64_t id) { return gtsam::Symbol('b', id); }
    inline gtsam::Symbol sym_landmark(uint64_t id) { return gtsam::Symbol('l', id); }

    fixed_lag_smoother::fixed_lag_smoother(const config& cfg) : _config(cfg) {
        // 1. Configure ISAM2 parameters used internally
        gtsam::ISAM2Params isam_params = {};
        isam_params.relinearizeThreshold = cfg.backend.relinearize_threshold;
        isam_params.relinearizeSkip = cfg.backend.relinearize_skip;

        // 2. Initialize smoother
        // Lag time is in seconds. The smoother automatically identifies keys to marginalize based on timestamps provided during update().
        smoother = std::make_unique<gtsam::IncrementalFixedLagSmoother>(cfg.backend.lag_time, isam_params);

        // 3. Calibration (legacy API support)
        calibration = boost::make_shared<gtsam::Cal3_S2>(cfg.camera.fx, cfg.camera.fy, 0.0, cfg.camera.cx, cfg.camera.cy);

        // 4. Initialize noise models
        visual_noise = gtsam::noiseModel::Isotropic::Sigma(2, 1.0); // ~1 pixel error
        velocity_noise = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector3::Constant(0.1));
        pose_noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.01, 0.01, 0.01, 0.05, 0.05, 0.05).finished());
        bias_noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.01, 0.01, 0.01).finished());
        robust_visual_noise = gtsam::noiseModel::Robust::Create(gtsam::noiseModel::mEstimator::Huber::Create(1.345), visual_noise);

        // Bias random walk (process noise)
        // config.imu stores continuous-time noise densities: sigma / sqrt(Hz)
        // We need discrete-time std dev for the BetweenFactor: sigma_d = sigma_c * sqrt(dt)
        // We assume a nominal keyframe spacing of 10Hz for defining the model, as the exact dt varies per frames.
        const double dt = 0.1;
        const double sigma_accel_rw = cfg.imu.accel_random_walk * std::sqrt(dt);
        const double sigma_gyro_rw = cfg.imu.gyro_random_walk * std::sqrt(dt);
        bias_rw_noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << sigma_accel_rw, sigma_accel_rw, sigma_accel_rw, sigma_gyro_rw, sigma_gyro_rw, sigma_gyro_rw).finished());
    }

    void fixed_lag_smoother::initialize(const std::shared_ptr<keyframe>& kf, const state& initial_state) {
        std::lock_guard<std::mutex> lock(mutex);

        // 1. Add priors
        new_factors.add(gtsam::PriorFactor<gtsam::Pose3>(sym_pose(kf->id), gtsam::Pose3(initial_state.pose.matrix()), pose_noise));
        new_factors.add(gtsam::PriorFactor<gtsam::Vector3>(sym_vel(kf->id), initial_state.velocity, velocity_noise));
        new_factors.add(gtsam::PriorFactor<gtsam::imuBias::ConstantBias>(sym_bias(kf->id), gtsam::imuBias::ConstantBias(initial_state.bias.accelerometer, initial_state.bias.gyroscope), bias_noise));
    
        // 2. Add initial values
        new_values.insert(sym_pose(kf->id), gtsam::Pose3(initial_state.pose.matrix()));
        new_values.insert(sym_vel(kf->id), initial_state.velocity);
        new_values.insert(sym_bias(kf->id), gtsam::imuBias::ConstantBias(initial_state.bias.accelerometer, initial_state.bias.gyroscope));

        // 3. Register timestamps
        new_timestamps[sym_pose(kf->id)] = kf->_timestamp;
        new_timestamps[sym_vel(kf->id)] = kf->_timestamp;
        new_timestamps[sym_bias(kf->id)] = kf->_timestamp;

        latest_state = initial_state;
    }

    void fixed_lag_smoother::add_keyframe(const std::shared_ptr<keyframe>& kf, const gtsam::PreintegratedCombinedMeasurements& imu_meas, const uint64_t prev_kf_id) {
        std::lock_guard<std::mutex> lock(mutex);

        // 1. IMU factor
        new_factors.add(gtsam::CombinedImuFactor(sym_pose(prev_kf_id), sym_vel(prev_kf_id), sym_pose(kf->id), sym_vel(kf->id), sym_bias(prev_kf_id), sym_bias(kf->id), imu_meas));

        // 2. Bias random walk
        new_factors.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(sym_bias(prev_kf_id), sym_bias(kf->id), gtsam::imuBias::ConstantBias(), bias_rw_noise));

        // 3. Predict initial estimate
        const gtsam::NavState prev_state(gtsam::Pose3(latest_state.pose.matrix()), latest_state.velocity);
        const gtsam::imuBias::ConstantBias prev_bias(latest_state.bias.accelerometer, latest_state.bias.gyroscope);
        const gtsam::NavState prop_state = imu_meas.predict(prev_state, prev_bias);

        // 4. Insert values and timestamps
        new_values.insert(sym_pose(kf->id), prop_state.pose());
        new_values.insert(sym_vel(kf->id), prop_state.velocity());
        new_values.insert(sym_bias(kf->id), prev_bias);

        new_timestamps[sym_pose(kf->id)] = kf->_timestamp;
        new_timestamps[sym_vel(kf->id)] = kf->_timestamp;
        new_timestamps[sym_bias(kf->id)] = kf->_timestamp;

        const gtsam::Pose3 body_p_sensor(_config._extrinsics.t_cam_imu.inverse().matrix());

        // 5. Visual factors (landmarks)
        for (size_t i = 0; i < kf->keypoints.size(); ++i) {
            const auto& mp = kf->map_points[i];

            if (!mp || mp->is_bad)
                continue;

            // Check if landmark is new and underconstrained before adding the factor
            if (observed_landmarks.find(mp->id) == observed_landmarks.end()) {
                if (mp->get_observation_count() < 2)
                    continue; // Skip entirely — no factor, no value

                new_values.insert(sym_landmark(mp->id), gtsam::Point3(mp->position));
                new_timestamps[sym_landmark(mp->id)] = kf->_timestamp;
                observed_landmarks.insert(mp->id);
            }

            const gtsam::Point2 measurement(kf->keypoints[i].pt.x, kf->keypoints[i].pt.y);
            new_factors.add(gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>(measurement, robust_visual_noise, sym_pose(kf->id), sym_landmark(mp->id), calibration, body_p_sensor));
        }

        // Update cache
        latest_state._timestamp = kf->_timestamp;
        latest_state.pose = se3(prop_state.pose().rotation().matrix(), prop_state.pose().translation());
        latest_state.velocity = prop_state.velocity();

        // Bias remains prev_bias until optimization updates it.
    }

    void fixed_lag_smoother::add_pose_prior(const uint64_t kf_id, const se3& pose) {
        std::lock_guard<std::mutex> lock(mutex);
        // Add a strong prior to pull the graph to the loop-corrected pose.
        const auto noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.01, 0.01, 0.01, 0.02, 0.02, 0.02).finished());
        new_factors.add(gtsam::PriorFactor<gtsam::Pose3>(sym_pose(kf_id), gtsam::Pose3(pose.matrix()), noise));
    }

    std::vector<uint64_t> fixed_lag_smoother::optimize() {
        std::lock_guard<std::mutex> lock(mutex);

        std::vector<uint64_t> marginalized_kf_ids;

        // 1. Capture keys before update to detect marginalization.
        std::set<uint64_t> pose_keys_before;
        for (const auto& [key, timestamp] : smoother->timestamps()) {
            const gtsam::Symbol sym(key);
            if (sym.chr() == 'x') // Pose keys only
                pose_keys_before.insert(sym.index());
        }
        
        // 2. Update the smoother
        try {
            smoother->update(new_factors, new_values, new_timestamps);
        }
        catch (const gtsam::IndeterminantLinearSystemException& e) {
            // Often happens if system is under-constrained (e.g., waiting for IMU bias convergence), no big deal.
            std::cerr << "Smoother Indeterminant: " << e.what() << std::endl;
        }

        // 3. Clear buffers
        new_factors.resize(0);
        new_values.clear();
        new_timestamps.clear();

        // 4. Detect marginalized pose keys
        const auto& current_timestamps = smoother->timestamps();
        for (const uint64_t kf_id : pose_keys_before) {
            const gtsam::Key key = gtsam::Symbol('x', kf_id);
            if (current_timestamps.find(key) == current_timestamps.end())
                marginalized_kf_ids.push_back(kf_id);
        }

        // 5. Clean up marginalized landmarks from internal tracking.
        for (auto it = observed_landmarks.begin(); it != observed_landmarks.end();)
            if (smoother->timestamps().find(gtsam::Symbol('l', *it)) == smoother->timestamps().end())
                it = observed_landmarks.erase(it);
            else
                ++it;

        // 6. Update latest state from optimized values
        if (!marginalized_kf_ids.empty() || !pose_keys_before.empty()) {
            // Find the most recent pose key still in the smoother.
            const gtsam::Values estimate = smoother->calculateEstimate();

            double latest_timestamp = -1.0;
            uint64_t latest_kf_id = 0;

            for (const auto& [key, timestamp] : current_timestamps) {
                const gtsam::Symbol sym(key);
                if (sym.chr() == 'x' && timestamp > latest_timestamp) {
                    latest_timestamp = timestamp;
                    latest_kf_id = sym.index();
                }
            }

            if (latest_timestamp >= 0.0 && estimate.exists(gtsam::Symbol('x', latest_kf_id))) {
                const gtsam::Pose3 p = estimate.at<gtsam::Pose3>(gtsam::Symbol('x', latest_kf_id));
                const gtsam::Vector3 v = estimate.at<gtsam::Vector3>(gtsam::Symbol('v', latest_kf_id));
                const gtsam::imuBias::ConstantBias b = estimate.at<gtsam::imuBias::ConstantBias>(gtsam::Symbol('b', latest_kf_id));

                latest_state.pose = se3(p.rotation().matrix(), p.translation());
                latest_state.bias.accelerometer = b.accelerometer();
                latest_state.bias.gyroscope = b.gyroscope();
                latest_state._timestamp = latest_timestamp;
                latest_state.velocity = v;
            }
        }

        return marginalized_kf_ids;
    }

    state fixed_lag_smoother::get_latest_state() const {
        std::lock_guard<std::mutex> lock(mutex);
        // If we want the optimized latest state, we should query the smoother.
        // However, the smoother might not have the very latest keyframe if optimize() hasn't run yet.
        // We return the cached state which "Prediction" before optimize, and "Optimized" after.
        return latest_state;
    }

    void fixed_lag_smoother::update_keyframe_state(std::shared_ptr<keyframe>& kf) {
        if (!kf)
            return;

        std::lock_guard<std::mutex> lock(mutex);

        // Calculate estimate
        const gtsam::Values estimate = smoother->calculateEstimate();

        // Note: calculateEstimate() can be slow if doing the full graph.
        // In ISAM2/FixedLag, we usually read from the Theta (current estimate) structure.
        if (estimate.exists(sym_pose(kf->id))) {
            const gtsam::Pose3 p = estimate.at<gtsam::Pose3>(sym_pose(kf->id));
            const gtsam::Vector3 v = estimate.at<gtsam::Vector3>(sym_vel(kf->id));
            const gtsam::imuBias::ConstantBias b = estimate.at<gtsam::imuBias::ConstantBias>(sym_bias(kf->id));

            // Update keyframe
            kf->set_pose(p);

            // Update internal cache if this is the latest frame.
            if (kf->_timestamp >= latest_state._timestamp) {
                latest_state.pose = se3(p.rotation().matrix(), p.translation());
                latest_state.bias.accelerometer = b.accelerometer();
                latest_state.bias.gyroscope = b.gyroscope();
                latest_state.velocity = v;
            }
        }
    }

} // namespace caai_slam