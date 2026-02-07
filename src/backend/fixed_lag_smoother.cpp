#include "caai_slam/backend/fixed_lag_smoother.hpp"

#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/inference/Symbol.h>

#include <iostream>

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
        smoother = std::make_unique<gtsam::IncrementalFixedLagSmoother>(cfg.backend.lag_time, isam_params);

        // 3. Calibration (legacy API support)
        calibration = boost::make_shared<gtsam::Cal3_S2>(cfg.camera.fx, cfg.camera.fy, 0.0, cfg.camera.cx, cfg.camera.cy);

        // 4. Initialize noise models
        // FIX: Relaxed visual noise from 0.5 to 1.5 pixels — AKAZE localization + triangulation
        // error easily exceeds 0.5px, and tight noise amplifies the information contribution
        // of poorly-triangulated landmarks relative to IMU, destabilizing the Hessian.
        visual_noise = gtsam::noiseModel::Isotropic::Sigma(2, 1.5);
        velocity_noise = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector3::Constant(0.1));
        pose_noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.01, 0.01, 0.01, 0.05, 0.05, 0.05).finished());
        bias_noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.01, 0.01, 0.01).finished());
        robust_visual_noise = gtsam::noiseModel::Robust::Create(gtsam::noiseModel::mEstimator::Huber::Create(1.345), visual_noise);

        // NOTE: bias_rw_noise is no longer needed — CombinedImuFactor handles bias evolution
        // internally via biasAccCovariance and biasOmegaCovariance set in imu_preintegration.cpp.
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
        // CombinedImuFactor connects: pose_i, vel_i, pose_j, vel_j, bias_i, bias_j
        // It internally models bias evolution (random walk) via biasAccCovariance and
        // biasOmegaCovariance parameters configured in imu_preintegration.cpp.
        new_factors.add(gtsam::CombinedImuFactor(sym_pose(prev_kf_id), sym_vel(prev_kf_id), sym_pose(kf->id), sym_vel(kf->id), sym_bias(prev_kf_id), sym_bias(kf->id), imu_meas));

        // FIX: REMOVED redundant BetweenFactor<imuBias::ConstantBias>.
        //
        // CombinedImuFactor ALREADY constrains bias_i <-> bias_j evolution. Adding a
        // separate BetweenFactor double-constrains the bias variables with conflicting
        // noise models. During marginalization, the Schur complement of these over-
        // constrained biases produces a near-singular Hessian — which is exactly the
        // IndeterminantLinearSystemException on frontal keys v90/v91/b90/b91.
        //
        // If you were using gtsam::ImuFactor (which does NOT model bias evolution),
        // then the BetweenFactor would be necessary. But with CombinedImuFactor, it
        // must be removed.

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
                new_factors.add(gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>(
                    uv, robust_visual_noise, sym_pose(kf->id), l_sym, calibration, body_p_sensor));
                
                // Update the timestamp to keep it in the smoothing window
                new_timestamps[l_sym] = kf->_timestamp;
                continue;
            }

            // CASE B: New Landmark Initialization
            // FIX: Require at least 3 total observations before initializing
            if (mp->get_observation_count() < 3)
                continue;

            // Collect valid past observers that are currently active in the smoother
            std::vector<std::pair<uint64_t, gtsam::Point2>> valid_past_observers;
            for (const auto& [obs_kf, obs_idx] : mp->get_observations()) {
                if (obs_kf->id == kf->id) continue;
                
                if (smoother->timestamps().find(sym_pose(obs_kf->id)) != smoother->timestamps().end()) {
                    gtsam::Point2 past_uv(obs_kf->keypoints[obs_idx].pt.x, obs_kf->keypoints[obs_idx].pt.y);
                    valid_past_observers.emplace_back(obs_kf->id, past_uv);
                }
            }

            // FIX: Require >= 2 past anchored observers (matching graph_optimizer).
            // With only 1 past observer, when that anchor is marginalized the landmark
            // has only 1 remaining projection factor (2 constraints for 3 DOF = rank deficient).
            if (valid_past_observers.size() >= 2) {
                // FIX: Validate initial position before inserting into graph.
                // Check that the triangulated point reprojects reasonably into the current frame.
                const gtsam::Pose3 current_pose = gtsam::Pose3(prop_state.pose().matrix());
                const gtsam::Point3 pt_world(mp->position);
                
                try {
                    const gtsam::PinholeCamera<gtsam::Cal3_S2> camera(current_pose * body_p_sensor, *calibration);
                    const gtsam::Point2 projected = camera.project(pt_world);
                    const double reproj_err = (projected - uv).norm();
                    
                    // Skip landmarks with large reprojection error (bad triangulation)
                    if (reproj_err > 10.0)
                        continue;
                } catch (...) {
                    continue; // Point behind camera or other projection failure
                }

                // 1. Initialize Landmark Value
                new_values.insert(l_sym, pt_world);
                new_timestamps[l_sym] = kf->_timestamp;
                observed_landmarks.insert(mp->id);

                // 2. Add factors for PAST observers
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
    }

    void fixed_lag_smoother::add_pose_prior(const uint64_t kf_id, const se3& pose) {
        std::lock_guard<std::mutex> lock(mutex);
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
            if (sym.chr() == 'x')
                pose_keys_before.insert(sym.index());
        }
        
        // 2. Update the smoother
        try {
            smoother->update(new_factors, new_values, new_timestamps);
        }
        catch (const gtsam::IndeterminantLinearSystemException& e) {
            std::cerr << "Smoother Indeterminant: " << e.what() << std::endl;
            std::cerr << "  Observed landmarks: " << observed_landmarks.size() << std::endl;
            std::cerr << "  New factors: " << new_factors.size() << std::endl;

            // FIX: Clear buffers to prevent stale factors from re-triggering on next call.
            new_factors.resize(0);
            new_values.clear();
            new_timestamps.clear();
            return {};
        }
        catch (const std::exception& e) {
            std::cerr << "Smoother exception: " << e.what() << std::endl;
            new_factors.resize(0);
            new_values.clear();
            new_timestamps.clear();
            return {};
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
        return latest_state;
    }

    void fixed_lag_smoother::update_keyframe_state(std::shared_ptr<keyframe>& kf) {
        if (!kf)
            return;

        std::lock_guard<std::mutex> lock(mutex);

        const gtsam::Values estimate = smoother->calculateEstimate();

        if (estimate.exists(sym_pose(kf->id))) {
            const gtsam::Pose3 p = estimate.at<gtsam::Pose3>(sym_pose(kf->id));
            const gtsam::Vector3 v = estimate.at<gtsam::Vector3>(sym_vel(kf->id));
            const gtsam::imuBias::ConstantBias b = estimate.at<gtsam::imuBias::ConstantBias>(sym_bias(kf->id));

            kf->set_pose(p);

            if (kf->_timestamp >= latest_state._timestamp) {
                latest_state.pose = se3(p.rotation().matrix(), p.translation());
                latest_state.bias.accelerometer = b.accelerometer();
                latest_state.bias.gyroscope = b.gyroscope();
                latest_state.velocity = v;
            }
        }
    }

} // namespace caai_slam
