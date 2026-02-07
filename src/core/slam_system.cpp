#include "caai_slam/core/slam_system.hpp"

#include "caai_slam/frontend/feature_extractor.hpp"
#include "caai_slam/backend/fixed_lag_smoother.hpp"
#include "caai_slam/mapping/keyframe_database.hpp"
#include "caai_slam/backend/graph_optimizer.hpp"
#include "caai_slam/vio/imu_preintegration.hpp"
#include "caai_slam/utils/triangulation.hpp"
#include "caai_slam/vio/visual_frontend.hpp"
#include "caai_slam/vio/vio_initializer.hpp"
#include "caai_slam/loop/loop_detector.hpp"
#include "caai_slam/mapping/local_map.hpp"
#include "caai_slam/utils/time_sync.hpp"

#include <iostream>

namespace caai_slam {
    slam_system::slam_system(const std::string& cfg_path) : status(system_status::NOT_INITIALIZED) {
        if (!_config.loadFromYAML(cfg_path))
            throw std::runtime_error("Failed to load configuration file: " + cfg_path);

        // Initialize shared resources
        _local_map = std::make_shared<local_map>(_config);
        _keyframe_database = std::make_shared<keyframe_database>();

        // Initialize subsystems
        _loop_closure_optimizer = std::make_unique<graph_optimizer>(_config);
        _time_sync = std::make_unique<time_sync>(_config._extrinsics.time_offset);
        _visual_frontend = std::make_unique<visual_frontend>(_config, _local_map);
        _fixed_lag_smoother = std::make_unique<fixed_lag_smoother>(_config);
        _vio_initializer = std::make_unique<vio_initializer>(_config);
        _feature_matcher = std::make_unique<feature_matcher>(_config);
        _loop_detector = std::make_unique<loop_detector>(_config);

        // Preintegrator initialized with zero bias initially, updated after init.
        _imu_preintegration = std::make_unique<imu_preintegration>(_config);

        // Initialize loop detector vocabulary
        _loop_detector->load_vocabulary("/home/gabriel/caai_slam/vocab/akaze_vocab.fbow");
    }

    slam_system::~slam_system() = default;

    void slam_system::process_initialization(const cv::Mat& image, const double ts) {
        se3 identity;
        const auto frame = _visual_frontend->process_image(image, ts, identity);

        _vio_initializer->add_frame(frame);

        state init_state;
        const initialization_status init_result = _vio_initializer->try_initialize(init_state);

        if (init_result != initialization_status::SUCCESS)
            return;

        std::cout << "System Initialized!" << std::endl;

        {
            std::lock_guard<std::mutex> lock(mutex);
            current_state = init_state;
            
            // FIX: Set last_imu_time to the initialization timestamp so the first
            // IMU measurement in TRACKING mode computes a correct dt instead of
            // using the fallback 0.005s. Without this, if there's a gap between
            // the last init IMU and the first tracking IMU, the first dt could be
            // incorrect (either the 5ms fallback or a very large value).
            last_imu_time = init_state._timestamp;
        }

        // 1. Initialize backend with first keyframe (priors)
        auto kf = std::make_shared<keyframe>(ts, gtsam::Pose3(init_state.pose.matrix()));
        kf->descriptors = frame->descriptors;
        kf->keypoints = frame->keypoints;
        kf->map_points.resize(kf->keypoints.size(), nullptr);
        
        _fixed_lag_smoother->initialize(kf, init_state);
        _local_map->add_keyframe(kf);
        last_keyframe = kf;

        _loop_closure_optimizer->add_first_keyframe(kf, init_state);

        // 2. Reset preintegration with estimated bias
        _imu_preintegration->reset(init_state.bias);

        status = system_status::TRACKING;
    }

    void slam_system::process_tracking(const cv::Mat& image, const double ts) {
        // 1. Predict pose using IMU
        state predicted_state;

        {
            std::lock_guard<std::mutex> lock(mutex);
            predicted_state = _imu_preintegration->predict(current_state);
        }

        // 2. Track visuals
        const auto curr_frame = _visual_frontend->process_image(image, ts, predicted_state.pose);

        // Update current state pose for visualization/output.
        {
            std::lock_guard<std::mutex> lock(mutex);
            current_state.velocity = predicted_state.velocity;
            current_state.pose = curr_frame->pose;
            current_state._timestamp = ts;
        }

        // 3. Keyframe decision
        if (_visual_frontend->need_new_keyframe(curr_frame, last_keyframe))
            create_and_insert_keyframe(curr_frame);
    }

    void slam_system::create_and_insert_keyframe(const std::shared_ptr<frame>& curr_frame) {
        if (!curr_frame)
            return;
        
        // 1. Create keyframe
        auto new_kf = std::make_shared<keyframe>(curr_frame->_timestamp, gtsam::Pose3(curr_frame->pose.matrix()));
        new_kf->descriptors = curr_frame->descriptors;
        new_kf->map_points = curr_frame->map_points;
        new_kf->keypoints = curr_frame->keypoints;

        if (new_kf->map_points.size() < new_kf->keypoints.size())
            new_kf->map_points.resize(new_kf->keypoints.size(), nullptr);

        if (last_keyframe) {
            if (last_keyframe->map_points.size() < last_keyframe->keypoints.size())
                last_keyframe->map_points.resize(last_keyframe->keypoints.size(), nullptr);

            std::vector<cv::DMatch> matches = _feature_matcher->match(new_kf->descriptors, last_keyframe->descriptors);

            const gtsam::Pose3 p_curr = new_kf->get_pose();
            const gtsam::Pose3 p_last = last_keyframe->get_pose();

            const se3 t_world_kf_curr(p_curr.rotation().matrix(), p_curr.translation());
            const se3 t_world_kf_last(p_last.rotation().matrix(), p_last.translation());

            const se3 t_cam_imu = _config._extrinsics.t_cam_imu;
            const se3 t_ic = t_cam_imu.inverse();

            const se3 pose_cam_curr = t_world_kf_curr * t_ic;
            const se3 pose_cam_last = t_world_kf_last * t_ic;

            for (const auto& m : matches) {
                if (m.queryIdx >= static_cast<int>(new_kf->keypoints.size()) || 
                    m.trainIdx >= static_cast<int>(last_keyframe->keypoints.size()))
                    continue;
                
                if (!new_kf->map_points[m.queryIdx] && !last_keyframe->map_points[m.trainIdx]) {
                    const double u0 = last_keyframe->keypoints[m.trainIdx].pt.x;
                    const double v0 = last_keyframe->keypoints[m.trainIdx].pt.y;
                    const double u1 = new_kf->keypoints[m.queryIdx].pt.x;
                    const double v1 = new_kf->keypoints[m.queryIdx].pt.y;

                    const vec2 norm0((u0-_config.camera.cx)/_config.camera.fx, (v0-_config.camera.cy)/_config.camera.fy);
                    const vec2 norm1((u1-_config.camera.cx)/_config.camera.fx, (v1-_config.camera.cy)/_config.camera.fy);

                    vec3 pt_world;
                    if (triangulate_dlt(pose_cam_last, norm0, pose_cam_curr, norm1, pt_world)) {
                        if (compute_parallax(pose_cam_last, pose_cam_curr, pt_world) < _config.frontend.parallax_min)
                            continue;

                        const vec3 p_cam = pose_cam_curr.inverse() * pt_world;
                        if (p_cam.z() < 0.1 || p_cam.z() > 50.0)
                            continue;

                        auto mp = std::make_shared<map_point>(pt_world, new_kf->descriptors.row(m.queryIdx));
                        {
                            std::lock_guard<std::mutex> last_kf_lock(last_keyframe->mutex);
                            mp->add_observation(last_keyframe, m.trainIdx);
                            last_keyframe->map_points[m.trainIdx] = mp;
                        }

                        mp->add_observation(new_kf, m.queryIdx);
                        new_kf->map_points[m.queryIdx] = mp;
                        _local_map->add_map_point(mp);
                    }
                }
                else if (!new_kf->map_points[m.queryIdx] && last_keyframe->map_points[m.trainIdx]) {
                    auto mp = last_keyframe->map_points[m.trainIdx];
                    new_kf->map_points[m.queryIdx] = mp;
                    mp->add_observation(new_kf, m.queryIdx);
                }
            }
        }

        // 2. Add to maps
        _local_map->add_keyframe(new_kf);
        _keyframe_database->add(new_kf);

        imu_bias bias_to_reset;
        {
            std::lock_guard<std::mutex> lock(mutex);
            bias_to_reset = current_state.bias;
        }

        // 3. Get preintegrated measurements since last keyframe
        const auto imu_factors = _imu_preintegration->get_and_reset(bias_to_reset);

        // FIX: Guard against degenerate preintegration.
        // If deltaTij is near-zero, the CombinedImuFactor will have a near-singular
        // covariance (infinite information) because no time has elapsed. This creates
        // an extremely tight constraint that dominates the Hessian and causes
        // IndeterminantLinearSystemException.
        if (imu_factors.deltaTij() < 1e-4) {
            std::cerr << "[System] WARNING: Skipping keyframe insertion — degenerate "
                      << "IMU preintegration (deltaTij=" << imu_factors.deltaTij() 
                      << "s). This keyframe would create a singular factor." << std::endl;
            
            // Don't add to backend — the IMU factor would be degenerate.
            // Reset preintegration and keep accumulating.
            _imu_preintegration->reset(bias_to_reset);
            
            // Remove keyframe from maps since we're not adding it to the backend
            // (leaving it in local_map without backend would cause inconsistency)
            // Note: This is a rare edge case.
            last_keyframe = new_kf; // Still update tracking reference
            return;
        }

        _fixed_lag_smoother->add_keyframe(new_kf, imu_factors, last_keyframe->id);
        _loop_closure_optimizer->add_keyframe(new_kf, imu_factors, last_keyframe->id);      

        // Optimize
        const auto marginalized = _fixed_lag_smoother->optimize();

        // Update local map (prune old)
        _local_map->prune_old_keyframes(curr_frame->_timestamp);

        // Update states from backend result
        _fixed_lag_smoother->update_keyframe_state(new_kf);

        {
            std::lock_guard<std::mutex> lock(mutex);
            current_state = _fixed_lag_smoother->get_latest_state();
            _imu_preintegration->reset(current_state.bias);
        }

        // 4. Loop closure check
        _loop_detector->add_keyframe(new_kf);
        const auto loop_res = _loop_detector->detect_loop(new_kf, _local_map->get_all_keyframes());

        if (loop_res.is_detected) {
            std::cout << "Loop detected between KF " << new_kf->id << " and KF " << loop_res.match_kf->id << std::endl;
            
            _loop_closure_optimizer->add_loop_constraint(new_kf->id, loop_res.match_kf->id, loop_res.t_match_query);
            correct_loop_closure(new_kf);
        }

        // 5. Update pointers
        last_keyframe = new_kf;
    }

    void slam_system::process_imu(const imu_measurement& meas) {
        // Synchronize timestamp
        imu_measurement synced = meas;
        synced._timestamp = _time_sync->imu_to_cam(meas._timestamp);

        if (status == system_status::NOT_INITIALIZED || status == system_status::INITIALIZING) {
            _vio_initializer->add_imu(synced);
            
            // FIX: Also track last_imu_time during initialization so the transition
            // to TRACKING doesn't have a stale timestamp. This prevents incorrect dt
            // on the first few IMU measurements after init completes.
            std::lock_guard<std::mutex> lock(mutex);
            last_imu_time = synced._timestamp;
        }
        else if (status == system_status::TRACKING) {
            double dt = 0.0;
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (last_imu_time < 0.0) {
                    // FIX: Use nominal dt from IMU frequency instead of hardcoded 0.005
                    dt = 1.0 / _config.imu.frequency;
                } else {
                    dt = synced._timestamp - last_imu_time;
                }
                last_imu_time = synced._timestamp;
            }
            
            // FIX: Validate dt before integrating.
            // Negative dt indicates timestamp regression (bad data or sync issue).
            // Very large dt indicates dropped packets or timestamp discontinuity.
            if (dt > 0.0 && dt < 0.5)
                _imu_preintegration->integrate(synced, dt);
            else if (dt < 0.0)
                std::cerr << "[IMU] WARNING: Negative dt=" << dt 
                          << "s — timestamp regression detected" << std::endl;
        }
    }

    void slam_system::process_image(const cv::Mat& image, const double ts) {
        system_status curr_status = status.load();
        
        if (curr_status == system_status::NOT_INITIALIZED) {
            status = system_status::INITIALIZING;
            curr_status = system_status::INITIALIZING;
        }

        if (curr_status == system_status::INITIALIZING)
            process_initialization(image, ts);
        else if (curr_status == system_status::TRACKING)
            process_tracking(image, ts);
    }

    void slam_system::correct_loop_closure(std::shared_ptr<keyframe>& curr_kf) {
        _loop_closure_optimizer->optimize(curr_kf, _local_map->get_all_keyframes());

        const se3 corrected_pose(curr_kf->get_pose().rotation().matrix(), curr_kf->get_pose().translation());
        _fixed_lag_smoother->add_pose_prior(curr_kf->id, corrected_pose);
        {
            std::lock_guard<std::mutex> lock(mutex);
            current_state.pose = se3(curr_kf->get_pose().rotation().matrix(), curr_kf->get_pose().translation());
        }

        std::cout << "Loop closure correction applied" << std::endl;
    }

    se3 slam_system::get_current_pose() const {
        std::lock_guard<std::mutex> lock(mutex);
        return current_state.pose;
    }

    state slam_system::get_current_state() const {
        std::lock_guard<std::mutex> lock(mutex);
        return current_state;
    }

    system_status slam_system::get_status() const {
        return status;
    }

    void slam_system::reset() {
        std::lock_guard<std::mutex> lock(mutex);

        std::cout << "[System] Resetting CAAI-SLAM..." << std::endl;

        status = system_status::NOT_INITIALIZED;

        last_keyframe.reset();
        last_image_timestamp = -1.0;
        last_imu_time = -1.0;

        _keyframe_database->clear();
        _vio_initializer->reset();
        _loop_detector->reset();

        _local_map = std::make_shared<local_map>(_config);
        _visual_frontend = std::make_unique<visual_frontend>(_config, _local_map);
        _fixed_lag_smoother = std::make_unique<fixed_lag_smoother>(_config);
        _imu_preintegration = std::make_unique<imu_preintegration>(_config);
        _loop_closure_optimizer = std::make_unique<graph_optimizer>(_config);

        std::cout << "[System] Reset complete. Waiting for initialization..." << std::endl;
    }

} // namespace caai_slam
