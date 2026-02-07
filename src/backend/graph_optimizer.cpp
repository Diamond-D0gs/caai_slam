#include "caai_slam/backend/graph_optimizer.hpp"

#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/inference/Symbol.h>

namespace caai_slam {
    // Helper symbol generators
    inline gtsam::Symbol sym_pose(uint64_t id) { return gtsam::Symbol('x', id); }
    inline gtsam::Symbol sym_vel(uint64_t id) { return gtsam::Symbol('v', id); }
    inline gtsam::Symbol sym_bias(uint64_t id) { return gtsam::Symbol('b', id); }
    inline gtsam::Symbol sym_landmark(uint64_t id) { return gtsam::Symbol('l', id); }

    graph_optimizer::graph_optimizer(const config& config) : _config(config) {
        // 1. ISAM2 parameters
        gtsam::ISAM2Params params = {};
        params.relinearizeThreshold = config.backend.relinearize_threshold;
        params.relinearizeSkip = config.backend.relinearize_skip;
        isam = gtsam::ISAM2(params);

        // 2. Camera calibration
        camera_calibration = boost::make_shared<gtsam::Cal3_S2>(config.camera.fx, config.camera.fy, 0.0 /* skew */, config.camera.cx, config.camera.cy);

        // 3. Noise models
        // Tighter priors for initialization.
        pose_noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.01, 0.01, 0.01, 0.05, 0.05, 0.05).finished());
        velocity_noise = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector3::Constant(0.1));
        bias_noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.01, 0.01, 0.01).finished());
    
        // Visual measurement noise (pixels)
        visual_noise = gtsam::noiseModel::Isotropic::Sigma(2, 1.0); // 1 pixel error
        robust_visual_noise = gtsam::noiseModel::Robust::Create(gtsam::noiseModel::mEstimator::Huber::Create(1.345), visual_noise);
    }

    void graph_optimizer::add_first_keyframe(const std::shared_ptr<keyframe>& kf, const state& initial_state) {
        std::lock_guard<std::mutex> lock(mutex);

        // 1. Add priors
        new_factors.add(gtsam::PriorFactor<gtsam::Pose3>(sym_pose(kf->id), gtsam::Pose3(initial_state.pose.matrix()), pose_noise));
        new_factors.add(gtsam::PriorFactor<gtsam::Vector3>(sym_vel(kf->id), initial_state.velocity, velocity_noise));
        new_factors.add(gtsam::PriorFactor<gtsam::imuBias::ConstantBias>(sym_bias(kf->id), gtsam::imuBias::ConstantBias(initial_state.bias.accelerometer, initial_state.bias.gyroscope), bias_noise));

        // 2. Add initial values
        new_values.insert(sym_pose(kf->id), gtsam::Pose3(initial_state.pose.matrix()));
        new_values.insert(sym_vel(kf->id), initial_state.velocity);
        new_values.insert(sym_bias(kf->id), gtsam::imuBias::ConstantBias(initial_state.bias.accelerometer, initial_state.bias.gyroscope));

        latest_state = initial_state;
    }

    void graph_optimizer::add_keyframe(const std::shared_ptr<keyframe>& kf, const gtsam::PreintegratedCombinedMeasurements& preintegrated_imu, uint64_t previous_kf_id) {
        std::lock_guard<std::mutex> lock(mutex);

        // 1. Add IMU factor (links previous to current).
        const gtsam::CombinedImuFactor imu_factor(sym_pose(previous_kf_id), sym_vel(previous_kf_id), sym_pose(kf->id), sym_vel(kf->id), sym_bias(previous_kf_id), sym_bias(kf->id), preintegrated_imu);
        new_factors.add(imu_factor);

        // 2. Add bias random walk factor (links previous bias to current bias).
        // Assumes bias does not change much between frames.
        // TODO: Load noise from config.
        const auto bias_rw_noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3).finished());
        new_factors.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(sym_bias(previous_kf_id), sym_bias(kf->id), gtsam::imuBias::ConstantBias(), bias_rw_noise));
        
        // 3. Predict initial estimate using IMU.
        const gtsam::NavState prev_state(gtsam::Pose3(latest_state.pose.matrix()), latest_state.velocity);
        const gtsam::imuBias::ConstantBias prev_bias(latest_state.bias.accelerometer, latest_state.bias.gyroscope);
        const gtsam::NavState prop_state = preintegrated_imu.predict(prev_state, prev_bias);

        // Insert initial values for current frame.
        new_values.insert(sym_pose(kf->id), prop_state.pose());
        new_values.insert(sym_vel(kf->id), prop_state.velocity());
        new_values.insert(sym_bias(kf->id), prev_bias);

        const gtsam::Pose3 body_p_sensor(_config._extrinsics.t_cam_imu.inverse().matrix());

        // 4. Add visual factors
        for (size_t i = 0; i < kf->keypoints.size(); ++i) {
            const auto& mp = kf->map_points[i];
            
            if (!mp || mp->is_bad)
                continue;

            // Check if landmark is new and underconstrained before adding the factor
            if (observed_landmarks.find(mp->id) == observed_landmarks.end()) {
                if (mp->get_observation_count() < 2)
                    continue; // Skip entirely — no factor, no value

                new_values.insert(sym_landmark(mp->id), gtsam::Point3(mp->position));
                observed_landmarks.insert(mp->id);
            }

            gtsam::Point2 measurement(kf->keypoints[i].pt.x, kf->keypoints[i].pt.y);

            // Add projection factor, Pose3 (camera pose in world) & Point3 (landmark in world).
            const gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2> vis_factor(measurement, robust_visual_noise, sym_pose(kf->id), sym_landmark(mp->id), camera_calibration, body_p_sensor);
            new_factors.add(vis_factor);
        }
    }

    void graph_optimizer::add_loop_constraint(const uint64_t kf_id_from, const uint64_t kf_id_to, const se3& rel_pose) {
        std::lock_guard<std::mutex> lock(mutex);

        // Noise for loop closure (usually derived from the matching confidence).
        const auto noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished()); // ~3 deg, ~10cm
        new_factors.add(gtsam::BetweenFactor<gtsam::Pose3>(sym_pose(kf_id_from), sym_pose(kf_id_to), gtsam::Pose3(rel_pose.matrix()), noise));
    
        pending_loop_closure = true;
    }

    Eigen::Matrix<double, 15, 15> graph_optimizer::compute_state_covariance(const uint64_t kf_id) {
        // State ordering: [rotation(3), position(3), velocity(3), bias_gyro(3), bias_accel(3)].
        Eigen::Matrix<double, 15, 15> covariance = Eigen::Matrix<double, 15, 15>::Identity();

        const gtsam::Key pose_key = sym_pose(kf_id);
        const gtsam::Key vel_key = sym_vel(kf_id);
        const gtsam::Key bias_key = sym_bias(kf_id);

        // TODO: Make more robust to marginalization failure.
        // Extract marginal covariances for each component.
        const gtsam::Matrix pose_cov = isam.marginalCovariance(pose_key);
        const gtsam::Matrix vel_cov = isam.marginalCovariance(vel_key);
        const gtsam::Matrix bias_cov = isam.marginalCovariance(bias_key);

        // Assemble block diagonal
        // GTSAM Pose3 covariance ordering: [rotation(3), translation(3)].
        covariance.block<6, 6>(0, 0) = pose_cov;
        covariance.block<3, 3>(6, 6) = vel_cov;
        covariance.block<3, 3>(9, 9) = bias_cov.block<3, 3>(0, 0); // Gyro bias
        covariance.block<3, 3>(12, 12) = bias_cov.block<3, 3>(3, 3); // Accel bias

        return covariance;
    }

    state graph_optimizer::optimize(std::shared_ptr<keyframe>& curr_kf, const std::vector<std::shared_ptr<keyframe>>& active_kfs) {
        std::lock_guard<std::mutex> lock(mutex);

        // 1. Update ISAM2
        isam.update(new_factors, new_values);

        // Additional iterations after loop closure for better convergence.
        if (pending_loop_closure) {
            isam.update();
            isam.update(); // Two calls is a reasonable balance between latency and accuracy for real-time VIO.
            pending_loop_closure = false;
        }

        // Clear new factor and value buffers.
        new_factors.resize(0);
        new_values.clear();

        // 2. Compute estimate
        const gtsam::Values result = isam.calculateEstimate();

        // 3. Update current state cache
        if (result.exists(sym_pose(curr_kf->id))) {
            const gtsam::Pose3 p = result.at<gtsam::Pose3>(sym_pose(curr_kf->id));
            const gtsam::Vector3 v = result.at<gtsam::Vector3>(sym_vel(curr_kf->id));
            const gtsam::imuBias::ConstantBias b = result.at<gtsam::imuBias::ConstantBias>(sym_bias(curr_kf->id));

            latest_state.velocity = v;
            latest_state._timestamp = curr_kf->_timestamp;
            latest_state.pose = se3(p.rotation().matrix(), p.translation());

            latest_state.bias.accelerometer = b.accelerometer();
            latest_state.bias.gyroscope = b.gyroscope();

            // Compute 15x15 state covariance.
            latest_state.covariance = compute_state_covariance(curr_kf->id);
        }

        // 4. Update all active keyframes and their observed map points.
        std::unordered_set<std::shared_ptr<map_point>> map_points_to_update;

        for (const auto& kf : active_kfs) {
            if (!kf)
                continue;

            // Update keyframe pose
            if (result.exists(sym_pose(kf->id))) {
                const gtsam::Pose3 optimized_pose = result.at<gtsam::Pose3>(sym_pose(kf->id));
                kf->set_pose(optimized_pose);
            }

            // Copy over any unique map points from the keyframe into the update set.
            map_points_to_update.insert(kf->map_points.begin(), kf->map_points.end());
        }

        map_points_to_update.erase(nullptr);

        // Update map points observed by current keyframe.
        for (const auto& mp : map_points_to_update)
            if (!mp->is_bad && result.exists(sym_landmark(mp->id))) {
                const gtsam::Point3 optimized_pt = result.at<gtsam::Point3>(sym_landmark(mp->id));
                std::lock_guard<std::mutex> mp_lock(mp->mutex);
                mp->position = optimized_pt;
            }

        return latest_state;
    }

    state graph_optimizer::get_last_state() const {
        std::lock_guard<std::mutex> lock(mutex);
        return latest_state;
    }

} // namespace caai_slam