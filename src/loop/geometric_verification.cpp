#include "caai_slam/loop/geometric_verification.hpp"
#include "caai_slam/loop/common.hpp"

#include <opencv2/features2d.hpp>

#include <cmath>

namespace caai_slam {
    geometric_verification::geometric_verification(const config& cfg) : _config(cfg) {
        teaser::RobustRegistrationSolver::Params params = {};
        params.noise_bound = cfg.backend.loop_closure_noise_pos;
        params.rotation_cost_threshold = 1e-6;
        params.rotation_max_iterations = 100;
        params.rotation_gnc_factor = 1.4;
        params.estimate_scaling = false; // System is metric
        params.cbar2 = 1.0;

        solver = std::make_unique<teaser::RobustRegistrationSolver>(params);
    }

    geometric_verification::result geometric_verification::verify_3d_3d(const std::shared_ptr<keyframe>& query, const std::shared_ptr<keyframe>& match) {
        Eigen::Matrix<double, 3, Eigen::Dynamic> src_cloud, target_cloud;
        get_matched_points(_config.frontend.match_ratio_thresh, query, match, src_cloud, target_cloud);

        if (src_cloud.cols() < static_cast<int32_t>(_config.loop.min_matches_geom))
            return {};

        // Solve t such that target = t * src
        solver->solve(src_cloud, target_cloud);
        const auto solution = solver->getSolution();
        if (!solution.valid)
            return {};

        const uint32_t inliers = count_inliers(_config.backend.loop_closure_noise_pos, solution, src_cloud, target_cloud);
        if (inliers < static_cast<uint32_t>(_config.loop.min_matches_geom))
            return {};

        result res = {};
        res.t_query_match = se3(solution.rotation, solution.translation).inverse(); // Relative pose t_query_match
        res.inlier_count = inliers;
        res.success = true;

        return res;
    }

} // namespace caai_slam