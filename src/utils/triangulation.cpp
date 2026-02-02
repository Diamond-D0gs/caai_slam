#include "caai_slam/utils/triangulation.hpp"

#include <Eigen/SVD>

#include <cmath>
#include <algorithm>

namespace caai_slam {
    bool triangulate_dlt(const se3& pose_0, const vec2& point_0, const se3& pose_1, const vec2& point_1, vec3& out_point_world) {
        // 1. Compute projection matrices (3x4)
        // P = [r_cw | t_cw]. Since input is t_world_cam, we need t_cam_world
        const se3 t_c0_w = pose_0.inverse(), t_c1_w = pose_1.inverse();

        // Construct 3x4 projection matrices
        // Since we are using normalized coords, k is the identity
        Eigen::Matrix<double, 3, 4> p0, p1;
        p0.block<3, 3>(0, 0) = t_c0_w.rotation.toRotationMatrix();
        p1.block<3, 3>(0, 0) = t_c1_w.rotation.toRotationMatrix();
        p0.col(3) = t_c0_w.translation;
        p1.col(3) = t_c1_w.translation;

        // 2. Construct design matrix A for AX = 0
        // For each view: x = (P1 * x) / (P3 * X) -> x * (P3 * x) - (P1 * X) = 0
        // x*p3 - p0 = 0, y*p3 - p1 = 0
        Eigen::Matrix<double, 4, 4> a;
        // View 1
        a.row(0) = point_0.x() * p0.row(2) - p0.row(0); // u * P_row3 - P_row1
        a.row(1) = point_0.y() * p0.row(2) - p0.row(1); // v * P_row3 - P_row2
        // View 2
        a.row(2) = point_1.x() * p1.row(2) - p1.row(0);
        a.row(3) = point_1.y() * p1.row(2) - p1.row(1);

        // 3. Solve SVD
        // The solution is the eigenvector corresponding to the smallest eigenvalue (last column of V in A = UVD^T).
        const Eigen::JacobiSVD<Eigen::Matrix<double, 4, 4>> svd(a, Eigen::ComputeFullV);
        const vec4 x_homogeneous = svd.matrixV().col(3);

        // 4. Normalize homogeneous coordinate
        if (std::abs(x_homogeneous(3)) < 1e-6)
            return false; // Point at infinity

        const vec3 x_world = x_homogeneous.head<3>() / x_homogeneous(3);

        // 5. Check chirality (positive depth)
        // Transform the point to both camera frames and check if Z > 0.
        const vec3 x_c0 = t_c0_w * x_world, x_c1 = t_c1_w * x_world;
        if (x_c0.z() <= 0 || x_c1.z() <= 0)
            return false;

        out_point_world = x_world;

        return true;
    }

    double compute_parallax(const se3& pose_0, const se3& pose_1, const vec3& point_world) {
        // Vector from point to camera centers
        const vec3 vec_to_c_0 = pose_0.translation - point_world, vec_to_c_1 = pose_1.translation - point_world;
        const double norm_0 = vec_to_c_0.norm(), norm_1 = vec_to_c_1.norm();

        if (norm_0 < 1e-6 || norm_1 < 1e-6)
            return 0.0;

        // Clamp to valid range [-1, 1] for acos
        const double cos_angle = std::clamp(vec_to_c_0.dot(vec_to_c_1) / (norm_0 * norm_1), -1.0, 1.0);
        
        return std::acos(cos_angle) * 180.0 / M_PI;
    }

} // namespace caai_slam