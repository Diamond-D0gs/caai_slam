#include "caai_slam/backend/marginalization.hpp"

#include <gtsam/inference/Symbol.h>

#include <iostream>
#include <cmath>

namespace caai_slam {
    void marginalization::compute(const gtsam::NonlinearFactorGraph& graph, const gtsam::Values& values) {
        std::lock_guard<std::mutex> lock(mutex);

        try {
            marginals_computer = std::make_unique<gtsam::Marginals>(graph, values, gtsam::Marginals::QR);
            is_computed = true;
        }
        catch (const std::exception& e) {
            std::cerr << "Marginalization: Failed to compute marginals: " << e.what() << std::endl;
            is_computed = false;
        }
    }

    mat6 marginalization::get_pose_covariance(const uint64_t kf_id) const {
        std::lock_guard<std::mutex> lock(mutex);

        if (!is_computed || !marginals_computer)
            return mat6::Identity(); // Return identity if failed.

        try {
            return marginals_computer->marginalCovariance(gtsam::Symbol('x', kf_id));
        }
        catch (...) {
            return mat6::Identity();
        }
    }

    mat3 marginalization::get_landmark_covariance(const uint64_t mp_id) const {
        std::lock_guard<std::mutex> lock(mutex);

        if (!is_computed || !marginals_computer)
            return mat3::Identity();

        try {
            return marginals_computer->marginalCovariance(gtsam::Symbol('l', mp_id));
        }
        catch (...) {
            return mat3::Identity();
        }
    }

    mat3 marginalization::get_velocity_covariance(const uint64_t kf_id) const {
        std::lock_guard<std::mutex> lock(mutex);

        if (!is_computed || !marginals_computer)
            return mat3::Identity();

        try {
            return marginals_computer->marginalCovariance(gtsam::Symbol('v', kf_id));
        }
        catch (...) {
            return mat3::Identity();
        }
    }

    double marginalization::get_pose_entropy(const uint64_t kf_id) const {
        std::lock_guard<std::mutex> lock(mutex);

        const mat6 cov = get_pose_covariance(kf_id);

        // Entropy H(x) = 0.5 * ln((2*pi*e)^k * det(Sigma))
        // For comparison purposes, we can just look at det(Sigma) or trace.
        // Determinant in sensitive to 0 eigenvalues.

        const double det = cov.determinant();

        if (det <= 0.0)
            return 0.0; // Avoid log(0)

        const int k = 6; // Dimensionality
        // Constant part: 0.5 * k * (1.0 + log(2 * pi))
        // Variable part: 0.5 * log(det)

        return 0.5 * std::log(det) + 0.5 * k * (1.0 + std::log(2 * M_PI));
    }

    void marginalization::clear() {
        std::lock_guard<std::mutex> lock(mutex);
        marginals_computer.reset();
        is_computed = false;
    }

} // namespace caai_slam