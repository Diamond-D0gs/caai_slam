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
        
    }

    void fixed_lag_smoother::initialize(const std::shared_ptr<keyframe>& kf, const state& initial_state) {
        std::lock_guard<std::mutex> lock(mutex);

        
    }

} // namespace caai_slam