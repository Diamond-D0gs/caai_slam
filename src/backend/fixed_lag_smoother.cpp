#include "caai_slam/backend/fixed_lag_smoother.hpp"

#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/inference/Symbol.h>

#include <iostream> // Added for improved logging

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
        // PATCH: Tightened visual noise from 1.0 to 0.5 to better balance with IMU
        visual_noise = gtsam::noiseModel::Isotropic::Sigma(2, 0.5); 
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
        // Use the actual dt from the measurement for accurate noise scaling
        double dt = imu_meas.deltaTij();
        if (dt < 1e-4) dt = 1e-4; // Avoid zero-division/invalid noise
        
        gtsam::Vector6 sigmas;
        sigmas << _config.imu.accel_random_walk * sqrt(dt), _config.imu.accel_random_walk * sqrt(dt), _config.imu.accel_random_walk * sqrt(dt),
                  _config.imu.gyro_random_walk * sqrt(dt), _config.imu.gyro_random_walk * sqrt(dt), _config.imu.gyro_random_walk * sqrt(dt);
        auto current_bias_rw = gtsam::noiseModel::Diagonal::Sigmas(sigmas);

        new_factors.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(sym_bias(prev_kf_id), sym_bias(kf->id), gtsam::imuBias::ConstantBias(), current_bias_rw));

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

            const gtsam::Symbol l_sym = sym_landmark(mp->id);
            const gtsam::Point2 uv(kf->keypoints[i].pt.x, kf->keypoints[i].pt.y);

            // CASE A: Landmark already exists in the smoother
            if (observed_landmarks.count(mp->id)) {
                // Add the new observation factor
                new_factors.add(gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>(
                    uv, robust_visual_noise, sym_pose(kf->id), l_sym, calibration, body_p_sensor));
                
                // IMPORTANT: Update the timestamp of the landmark to keep it in the smoothing window
                new_timestamps[l_sym] = kf->_timestamp;
                continue;
            }

            // CASE B: New Landmark Initialization
            // We need at least 2 observations total (1 past + 1 current) to triangulate/constrain
            if (mp->get_observation_count() < 2)
                continue;

            // Collect valid past observers that are currently active in the smoother
            std::vector<std::pair<uint64_t, gtsam::Point2>> valid_past_observers;
            for (const auto& [obs_kf, obs_idx] : mp->get_observations()) {
                if (obs_kf->id == kf->id) continue; // Skip current frame
                
                // Check if this observer is still in the smoother's active window
                if (smoother->timestamps().find(sym_pose(obs_kf->id)) != smoother->timestamps().end()) {
                    gtsam::Point2 past_uv(obs_kf->keypoints[obs_idx].pt.x, obs_kf->keypoints[obs_idx].pt.y);
                    valid_past_observers.emplace_back(obs_kf->id, past_uv);
                }
            }

            // If we have at least 1 valid past anchored observer, we can initialize
            if (valid_past_observers.size() >= 1) {
                // 1. Initialize Landmark Value (Point3)
                new_values.insert(l_sym, gtsam::Point3(mp->position));
                new_timestamps[l_sym] = kf->_timestamp;
                observed_landmarks.insert(mp->id);

                // 2. Add factors for PAST observers (anchors)
                for (const auto& [past_id, past_uv] : valid_past_observers) {
                    new_factors.add(gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>(
                        past_uv, robust_visual_noise, sym_pose(past_id), l_sym, calibration, body_p_sensor));
                }

                // 3. Add factor for CURRENT observer
                new_factors.add(gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>(
                    uv, robust_visual_noise, sym_pose(kf->id), l_sym, calibration, body_p_sensor));
            }
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
            // PATCH: Improved logging for debugging singularities
            std::cerr << "Smoother Indeterminant: " << e.what() << std::endl;
            std::cerr << "  Observed landmarks: " << observed_landmarks.size() << std::endl;
            std::cerr << "  New factors: " << new_factors.size() << std::endl;
            // Often happens if system is under-constrained (e.g., waiting for IMU bias convergence), no big deal.
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